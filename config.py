import collections
import torch.nn as nn

class Config(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, config_path='./config.cfg', sep=';', *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
        
        with open(config_path, 'r') as f:
            confs = {}
            for l in f.readlines():
                if (l[0] is not '#') and (l[0] is not '\n'): # remove comments and empty lines
                    splt = l.split(sep)
                    confs[splt[0]] = [eval(e) for e in splt[1:]]       
            self.update(confs)
  
    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key
    
    def __str__(self):
        return str(self.store)
    
    def __repr__(self):
        return repr(self.store)
    
    def create_multi_dict(self):
        """Used to create as much configuration needed to run experiments with
        all the possible combinations of values in the conf file."""
        prev_configs = [{}]
        for key, vals in self.store.items():
            new_configs = []
            for v in vals:
                for conf in prev_configs:
                    new_conf = {}
                    new_conf.update(conf)
                    new_configs.append(new_conf)
                    new_conf[key] = v   
                            
            prev_configs = new_configs
                
        return new_configs