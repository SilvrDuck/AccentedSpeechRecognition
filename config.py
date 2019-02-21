import collections
import torch.nn as nn

class Config(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, config_path='./default_config.cfg', sep=' ', *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
        
        with open(config_path, 'r') as f:
            confs = {}
            for l in f.readlines():
                if (l[0] is not '#') and (l[0] is not '\n'): # remove comments and empty lines
                    sep_idx = l.find(sep)
                    confs[l[:sep_idx]] = eval(l[sep_idx+1:])
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
        return self.store.__str__()
    
    def __repr__(self):
        return self.store.__repr__()

    
#    def create_multi_dict(self):
#        """ Not recomended, please use the patch_config method instead
#        """
#        """ Used to create as much configuration needed to run experiments with
#        all the possible combinations of values in the conf file."""
#        prev_configs = [{}]
#        for key, vals in self.store.items():
#            new_configs = []
#            for v in vals:
#                for conf in prev_configs:
#                    new_conf = {}
#                    new_conf.update(conf)
#                    new_configs.append(new_conf)
#                    new_conf[key] = v   
#                            
#            prev_configs = new_configs
#                
#        return new_configs
    
    def patch_config(self, patch_path, patch_sep='!', sep=' '):
        """Takes a file with config patches separated by a line 
        starting with the 'patch_sep' argument.
        For each creates a new config based on the default one."""
        
        new_configs = []
        
        with open(patch_path, 'r') as f:
            current = {}
            for l in f.readlines():
                if (l[0] is not '#') and (l[0] is not '\n'):
                    if (l[0] is '!'):
                        new_configs.append(current)
                        current = {}
                    else:
                        sep_idx = l.find(sep)
                        current[l[:sep_idx]] = eval(l[sep_idx+1:])
                        
            # Checks if last patch was added
            if len(current) > 0:
                new_configs.append(current)
                
        final_configs = [self.store.copy() for __ in range(len(new_configs))]
        [store.update(conf) for conf, store in zip(new_configs, final_configs)]
        
        return final_configs if len(final_configs) > 0 else self.store
            