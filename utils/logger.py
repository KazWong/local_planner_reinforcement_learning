import json
import joblib
import shutil
import numpy as np
import tensorflow.compat.v1 as tf
import time, os

tf.disable_v2_behavior()
tf.enable_resource_variables()

class EpochLogger(object):
    def __init__(self, output_dir=None, output_fname='progress.txt'):
        self.output_dir = output_dir
        if os.path.exists(self.output_dir):
            print("Warning: Log dir %s already exists!"%self.output_dir)
        else:
            os.makedirs(self.output_dir)
        #TODO: close the file when finish
        self.output_file = open(os.path.join(self.output_dir, output_fname), 'w')
        print("Logging data to %s"%self.output_file.name)
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}

    def log_tabular(self, key, val):
        #epoch, step, EpLen, EpRet, Loss,
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    def dump_tabular(self):
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-"*n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g"%val if hasattr(val, "__float__") else val
            print(fmt%(key, valstr))
            vals.append(val)
        print("-"*n_slashes)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers)+"\n")
            self.output_file.write("\t".join(map(str,vals))+"\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False

    def save_train_model(self, sess, itr=None):
        if not hasattr(self, 'train_saver'):
            self.train_saver = tf.train.Saver()
        fpath = 'train_save' + ('%d'%itr if itr is not None else '')
        fpath = os.path.join(self.output_dir, fpath)
        if os.path.exists(fpath):
            shutil.rmtree(fpath)
        save_path = self.train_saver.save(sess, os.path.join(fpath, 'model'))
        print("Model saved in path: %s" % save_path)

    def restore_model(self, sess, itr=None):
        if not hasattr(self, 'train_saver'):
            self.train_saver = tf.train.Saver()
        fpath = 'train_save' + ('%d'%itr if itr is not None else '')
        fpath = os.path.join(self.output_dir, fpath)
        print(fpath)
        self.train_saver.restore(sess, tf.train.latest_checkpoint(fpath))
        print("Model restored from path: %s" % fpath)

    def restore_weight(self, sess, var_list):
        fpath = 'train_save2'
        fpath = os.path.join(self.output_dir, fpath)
        print('restore weights: ', fpath)
        self.train_restore = tf.train.Saver(var_list)
        self.train_restore.restore(sess, tf.train.latest_checkpoint(fpath))

    def save_model_info(self, sess):
        tf.train.write_graph(sess.graph_def, self.output_dir, 'model_info.pbtxt', True)

    def save_config(self, config):
        output = json.dumps(config_json)
        with open(os.path.join(self.output_dir, "config.json"), 'w') as out:
            out.write(output)
