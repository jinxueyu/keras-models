
import os
from _keras_ernie import load_from_checkpoint

ernie_path = "/Users/xueyu/Workspace/embidding/paddle-hub/baidu/ernie/model-ernie1.0.1"
init_checkpoint = os.path.join(ernie_path, 'params')
ernie_config_path = os.path.join(ernie_path, 'ernie_config.json')
ernie_vocab_path = os.path.join(ernie_path, 'vocab.txt')
ernie_version = "model-1.0.1"

model = load_from_checkpoint(init_checkpoint, ernie_config_path, ernie_vocab_path, ernie_version,
            max_seq_len=128, num_labels=2, use_fp16=False, training=False, seq_len=None, name='ernie')
model.summary()


def build_ernie_model():
    import os
    from _keras_ernie import load_from_checkpoint
    ernie_path = "/root/ERNIE_stable-1.0.1"
    init_checkpoint = os.path.join(ernie_path, 'params')
    ernie_config_path = os.path.join(ernie_path, 'ernie_config.json')
    ernie_vocab_path = os.path.join(ernie_path, 'vocab.txt')
    ernie_version = "stable-1.0.1"

    model = load_from_checkpoint(init_checkpoint, ernie_config_path, ernie_vocab_path, ernie_version,
                                 max_seq_len=128, num_labels=2, use_fp16=False, training=False, seq_len=None,
                                 name='ernie')
    model.summary()
