from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from replace_progress_panel import validate_panel
from recover_owned_launch import cluster_helper_environment


class PanelReplacementTests(unittest.TestCase):
    def test_a_gpu_training_command_cannot_be_a_panel_victim(self):
        with self.assertRaises(ValueError):
            validate_panel(['python','-m','pear.trainer.main','--run','/same'],Path('/same'))

    def test_a_panel_for_another_run_cannot_be_replaced(self):
        with self.assertRaises(ValueError):
            validate_panel(['python','-m','pear.monitor.progress_panel','--run','/other'],Path('/same'))

    def test_matching_cpu_panel_is_eligible(self):
        validate_panel(['python','-m','pear.monitor.progress_panel','--run','/same'],Path('/same'))

    def test_cluster_helper_environment_removes_only_proxy_routing(self):
        source={'HTTP_PROXY':'corporate','https_proxy':'corporate','ALL_PROXY':'corporate',
                'NO_PROXY':'.byted.org','RAY_ADDRESS':'private:6379','PEAR_LAUNCH_ID':'fixture'}
        result=cluster_helper_environment(source)
        self.assertEqual(result,{'NO_PROXY':'.byted.org','RAY_ADDRESS':'private:6379','PEAR_LAUNCH_ID':'fixture'})
        self.assertEqual(source['HTTP_PROXY'],'corporate')


if __name__=='__main__':unittest.main()
