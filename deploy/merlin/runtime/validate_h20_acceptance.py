#!/usr/bin/env python3
"""Validate pinned H20 compatibility evidence while preserving upstream failures.

This is not a training-completion certificate. The reviewed profile pins the
fresh GPU harness, fixed plan, CUDA sources and unchanged upstream test files.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

HOPPER_TEST = 'tests/models/test_qwen3_5_rollout_kernels_hopper_on_gpu.py'
MIXED = 'test_a_mixed_step_matches_the_reference_eagerly_and_from_a_graph'
CHUNK = 'test_the_chunk_kernels_equal_flas'
BUCKETS = (32,48,64,80,96,112,128,160,176,208,224,240,256,272,288,336,368,384,432,480,
           496,512,592,624,640,656,672,688,720,768,784,800,816,848,864,880,912,928,944,960,976,1008,1024)
STATES = ([8,3,2560], [8,12,128,128])
FIXTURE_CASES = [
    {'name':'pinned_fixture_seed1','seed':1,'lengths':[1,1,1,150,64,37],'decodes':3,'steps':1,'reset_steps':[]},
    {'name':'pinned_fixture_seed7','seed':7,'lengths':[1,1,150,64,37],'decodes':2,'steps':1,'reset_steps':[]},
]
FIXTURE_INIT = {1:[1,1,1,1,1,0], 7:[1,1,1,1,0]}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def one(rows, kind):
    matches = [x for x in rows if x.get('kind') == kind]
    require(len(matches) == 1, f'Expected exactly one {kind} record')
    return matches[0]


def known_failure_text(text):
    """An unrelated assertion in the same test is not the documented exception."""
    require('Tensor-likes are not close!' in text, 'Known test failed for another reason')
    counts = set(re.findall(r'Mismatched elements:\s*(\d+)\s*/\s*(\d+)', text))
    require(counts == {('1','390144')}, 'Known test has a different mismatch count')
    for label, expected in (('absolute', 0.19236763856998773), ('relative', 0.17386333927835193)):
        values = re.findall(r'Greatest ' + label + r' difference:\s*([0-9.eE+-]+) at index \(([^)]+)\) \(up to ([0-9.eE+-]+) allowed\)', text)
        require(values and all(math.isclose(float(v), expected, rel_tol=1e-14, abs_tol=0) and
                               index.replace(' ','') == '77,0,30' and float(t) == .05 for v,index,t in values),
                f'Known test has different {label} error/coordinate/tolerance')


def validate_junit(profile, junit, suite_exit):
    root = ET.parse(junit).getroot()
    require(not list(root.iter('error')), 'Pytest setup/collection/runtime error')
    expected = profile['expected_tests']
    require(len(expected) == len(set(expected)) == 29, 'Profile does not contain 29 unique tests')
    observed, failed, skipped = {}, [], []
    for case in root.iter('testcase'):
        name = case.attrib['classname'].replace('.', '/') + '.py::' + case.attrib['name']
        require(name not in observed, f'Duplicate test result: {name}')
        failures, skips = case.findall('failure'), case.findall('skipped')
        require(len(failures) + len(skips) <= 1, f'Ambiguous test outcome: {name}')
        state = 'failed' if failures else 'skipped' if skips else 'passed'
        observed[name] = state
        if failures:
            failed.append(name)
            require(name == profile['documented_exception']['test'], f'Unexpected failure: {name}')
            known_failure_text(failures[0].get('message', '') + '\n' + ''.join(failures[0].itertext()))
        if skips:
            skipped.append(name)
            require(skips[0].get('message') == profile['expected_skip_reason'], f'Unexpected skip reason: {name}')
    require(set(observed) == set(expected), 'The complete 29-test upstream GPU suite was not observed')
    require(len(profile['expected_skipped']) == len(set(profile['expected_skipped'])) == 4 and
            set(skipped) == set(profile['expected_skipped']), 'Unexpected skipped tests')
    require(suite_exit == (1 if failed else 0), 'Pytest exit code differs from its report')
    for suite in root.iter('testsuite'):
        cases = list(suite.iter('testcase'))
        require(int(suite.get('errors', '0')) == 0, 'JUnit suite-level errors')
        if 'tests' in suite.attrib:
            require(int(suite.attrib['tests']) == len(cases), 'JUnit count differs from actual test records')
        for attribute, tag in (('failures','failure'), ('skipped','skipped')):
            if attribute in suite.attrib:
                require(int(suite.attrib[attribute]) == sum(c.find(tag) is not None for c in cases), f'JUnit {attribute} summary differs')
    return observed, failed


def expected_records(plan):
    cases = FIXTURE_CASES + plan['cases']
    require(len(plan['cases']) == 10 and sum(c['steps'] for c in plan['cases']) == 44,
            'Expected ten fixed cases with 44 updates, plus two fixtures')
    strict, begins, steps, ends = [], [], [], []
    def add(label, shape=None):
        row = {'kind':'strict_check','label':label,'passed':True}
        if shape is not None:
            row.update(shape=shape, bitwise_equal=True, all_finite=True)
        strict.append(row)
    for case in cases:
        name, decodes, lengths = case['name'], case['decodes'], case['lengths']
        tokens, count = sum(lengths), len(lengths)
        graph_rows = next(x for x in BUCKETS if x >= tokens) if tokens <= 1024 else None
        if graph_rows is not None:
            for i, shape in enumerate(STATES):
                add(f'{name}/empty_capture_state{i}', shape)
        begins.append({'kind':'case_begin','case':case,'tokens_per_step':tokens,'graph_rows':graph_rows})
        for step in range(case['steps']):
            prefix = f'{name}/step{step}'
            add(prefix+'/g_cum', [tokens-decodes,12])
            for key in ('w','u','o'):
                add(prefix+'/'+key, [tokens-decodes,12,128])
            add(prefix+'/final', [count-decodes,12,128,128])
            add(prefix+'/all_output_finite')
            if name == 'pinned_fixture_seed1':
                add(prefix+'/measured_custom_vs_FLA_gated', [tokens,12,128])
                add(prefix+'/known_shared_fp64_failure_preserved')
            if graph_rows is not None:
                add(prefix+'/graph/output', [tokens,12,128])
                add(prefix+'/graph/padding_zero')
                for i, shape in enumerate(STATES):
                    add(prefix+f'/graph/state{i}', shape)
            init = [1]*count
            if name.startswith('pinned_fixture_'):
                init = FIXTURE_INIT[case['seed']]
            elif step == 0:
                for i in range(decodes,count):
                    init[i] = int((i-decodes)%2 == 0)
                if count == 1:
                    init[0] = 0
            if step in case['reset_steps']:
                init[-1] = 0
            steps.append({'kind':'case_step_complete','case':name,'step':step,'init':init,'reset':step in case['reset_steps']})
        ends.append({'kind':'case_complete','case':name,'steps':case['steps']})
    return strict, begins, steps, ends


def validate_baseline(report, checks, profile):
    baselines = report.get('measured_fp64_baselines')
    require(isinstance(baselines,list) and len(baselines) == 1, 'Missing fresh baseline measurement')
    baseline = baselines[0]
    require(one(checks,'measured_fp64_shared_baseline') == {'kind':'measured_fp64_shared_baseline', **baseline},
            'Report baseline differs from the actual measurement log')
    require(baseline.get('fixture_seed') == 1 and baseline.get('measured_shared_failure') is True and
            baseline.get('same_gated_output_bitexact') is True, 'Fresh custom/FLA baseline comparison failed')
    comparisons = baseline.get('comparisons', [])
    require(len(comparisons) == 2 and {x['path'] for x in comparisons} == {'custom','FLA'}, 'Baseline needs exactly two independent reference comparisons')
    known = profile['documented_exception']
    expected_failure = {'index':[77,0,30], 'actual':0.9140625, 'expected':1.1064301385699877}
    require(known['bad_coordinates'] == [expected_failure['index']] and known['actual'] == expected_failure['actual'] and
            known['reference'] == expected_failure['expected'] and known['fp64_rtol'] == known['fp64_atol'] == .05,
            'Baseline profile differs from the reviewed exception')
    for comparison in comparisons:
        require(comparison.get('rtol') == comparison.get('atol') == .05 and
                type(comparison.get('failure_count')) is int and comparison['failure_count'] == 1 and
                comparison.get('assertion_passed') is False and comparison.get('all_finite') is True and
                comparison.get('failures') == [expected_failure], 'Baseline has unapproved numerical failures')
        known_failure_text(comparison.get('error',''))
    require(report.get('baseline_verified') is True and report.get('known_fp64_shared_baseline_retained') is True,
            'Harness did not verify the fresh shared baseline')
    return baseline


def validate(profile_path, repo, junit, regression_dir, harness, plan_path, suite_exit, regression_exit):
    profile = json.loads(profile_path.read_text())
    require(profile['profile'] == 'h20_fla_reference_compat_v1', 'Unexpected acceptance profile')
    require(profile['upstream_commit'] == 'f32aa1f7c2ec9c3f45630cc9f8541b3201e8e087', 'Unexpected upstream commit')
    require(sha(repo / profile['cuda_source']) == profile['cuda_source_sha256'], 'FMA fix differs from reviewed source')
    for name, expected in profile['original_test_source_sha256'].items():
        require(sha(repo/name) == expected, f'Original GPU test was edited: {name}')
    csrc = repo/Path(profile['cuda_source']).parent
    source_hashes = {p.name:sha(p) for p in csrc.iterdir() if p.is_file()}
    require(source_hashes == profile['csrc_sha256'], 'Deployed CUDA source differs from reviewed candidate')
    observed, failed = validate_junit(profile, junit, suite_exit)

    report_path, checks_path = regression_dir/'report.json', regression_dir/'checks.jsonl'
    report = json.loads(report_path.read_text())
    require(report.get('backend') == 'original', 'Final acceptance must load the effective deployed source')
    require(report.get('strict_regression_pass') is True and report.get('strict_failures') == [], 'Strict numerical regression failed/incomplete')
    require(not report.get('error') and report.get('full_preflight_pass') is False, 'Regression error or unsupported full-preflight claim')
    require(report.get('harness_sha256') == sha(harness) == profile['regression_harness_sha256'], 'Regression harness differs from the reviewed execution')
    plan = json.loads(plan_path.read_text())
    require(report['plan'] == plan and canonical_sha(plan) == profile['regression_plan_sha256'], 'Fixed regression case plan changed')
    checks = [json.loads(line) for line in checks_path.read_text().splitlines() if line.strip()]
    require(checks and checks[0]['kind'] == 'plan' and checks[-1] == {'kind':'report','report':report}, 'Regression log is incomplete or differs from report')
    require(one(checks,'report') == checks[-1], 'Multiple final reports')
    require(one(checks,'plan') == {'kind':'plan','plan':plan,'plan_sha256':canonical_sha(plan)}, 'Logged plan differs')
    allowed = {'plan','runtime_provenance','strict_check','case_begin','case_step_complete','case_complete',
               'unmodified_original_test','measured_fp64_shared_baseline','report'}
    require(all(x.get('kind') in allowed for x in checks), 'Unexpected execution/error/candidate-load record')
    strict, begins, steps, ends = expected_records(plan)
    require(len(strict) == profile['strict_check_count'] == 458 and
            sum('shape' in x for x in strict) == profile['exact_finite_check_count'] == 371, 'Profile check coverage differs')
    for kind, expected in (('strict_check',strict), ('case_begin',begins), ('case_step_complete',steps), ('case_complete',ends)):
        require([x for x in checks if x['kind'] == kind] == expected, f'Incomplete, duplicate or changed {kind} records')
    for row in checks:
        if row['kind'] == 'strict_check':
            require(row['passed'] is True and ('shape' not in row or
                    (row['bitwise_equal'] is True and row['all_finite'] is True)), 'Numeric booleans are not true')
        if row['kind'] == 'case_step_complete':
            require(type(row['reset']) is bool, 'Reset state is not a boolean')

    runtime = report.get('runtime_provenance', {})
    require(one(checks,'runtime_provenance') == {'kind':'runtime_provenance', **runtime}, 'Runtime provenance differs from report')
    require(runtime.get('backend') == 'original' and runtime.get('repo') == str(repo.resolve()) and
            runtime.get('native_source_dir') == str((repo/Path(profile['cuda_source']).parent).resolve()) and
            runtime.get('source_sha256') == runtime.get('repository_source_sha256') == source_hashes,
            'Fresh run did not use the deployed candidate source')
    for key in ('source_unchanged_after_run','source_unchanged_after_load','test_source_unchanged_after_run','library_unchanged_after_run'):
        require(runtime.get(key) is True, f'Runtime identity not stable: {key}')
    require(runtime.get('test_source_sha256') == profile['original_test_source_sha256'][HOPPER_TEST], 'Fresh run used different original tests')
    fixtures = {value[0]:value[1] for value in plan['pinned_original_fixtures'].values()}
    require(runtime.get('fixtures_sha256') == fixtures, 'Fresh run did not load both fixed input fixtures')
    library = Path(runtime.get('library',''))
    require(library.is_absolute() and library.name == 'pear_rollout_hopper.so' and
            re.fullmatch('[a-f0-9]{64}', runtime.get('library_sha256','')) and sha(library) == runtime['library_sha256'],
            'Actual original-backend extension artifact is missing or changed')

    baseline = validate_baseline(report, checks, profile)
    originals = report['original_tests']
    require(len(originals) == 2 and all(type(x['passed']) is bool for x in originals) and
            [(x['name'],x['passed']) for x in originals] == [(CHUNK,True),(MIXED,False)],
            'Unexpected original fixture assertion outcomes')
    known_failure_text(originals[1].get('error',''))
    require([x for x in checks if x['kind'] == 'unmodified_original_test'] ==
            [{'kind':'unmodified_original_test',**x} for x in originals], 'Original assertions differ from actual log')
    require(regression_exit == 1 and report['original_tests_all_pass'] is False,
            'The original failing assertion must remain reported as failed')
    return {
        'profile':profile['profile'], 'accepted':True, 'checked_at':time.time(),
        'original_suite_passed':not failed, 'original_suite':observed,
        'original_counts':{s:sum(x == s for x in observed.values()) for s in ('passed','failed','skipped')},
        'documented_exception':profile['documented_exception'] if failed else None,
        'strict_regression_passed':True, 'strict_check_count':len(strict), 'measured_fp64_baseline':baseline,
        'runtime_provenance':runtime, 'scope':profile['scope'], 'paper_metrics_reproduced':False,
        'grpo_validated_by_this_profile':False,
        'evidence_sha256':{'profile':sha(profile_path), 'junit':sha(junit), 'regression_report':sha(report_path),
                           'regression_checks':sha(checks_path), 'harness':sha(harness), 'plan':sha(plan_path)},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('profile','repo','junit','regression_dir','harness','plan','output'):
        p.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    p.add_argument('--suite-exit',type=int,required=True)
    p.add_argument('--regression-exit',type=int,required=True)
    a = p.parse_args()
    result = validate(a.profile,a.repo,a.junit,a.regression_dir,a.harness,a.plan,a.suite_exit,a.regression_exit)
    require(not a.output.exists(), 'Refusing to overwrite an acceptance result')
    temporary = a.output.with_suffix('.json.partial')
    with temporary.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    temporary.replace(a.output)
    print(json.dumps({'h20_compatibility_accepted':True,'original_suite_passed':result['original_suite_passed'],
                      'original_counts':result['original_counts'],'profile':result['profile']}))


if __name__ == '__main__':
    main()
