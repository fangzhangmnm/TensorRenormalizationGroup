import collections
import warnings
import hashlib

class PathFinder:
    def __init__(self, source_file, pars, ignore_pars=[], **kwargs):
        self.prefix = "data/%s/"%source_file.replace(".py", "", 16)
        if not isinstance(ignore_pars, collections.Iterable):
            ignore_pars = set((ignore_pars,))
        else:
            ignore_pars = set(ignore_pars)
        self.ignore_pars = ignore_pars
        self.pars = pars.copy()
        self.pars.update(kwargs)

    def generate_path(self, midfix, extension="", **kwargs):
        pars = self.pars.copy()
        pars.update(kwargs)
        postfix = type(self).dict_to_postfix(pars, self.ignore_pars)
        name = self.prefix + midfix + "_-_" + postfix + extension
        if len(name)>50:
            # Hash the postfix if the name is too long
            bpostfix = postfix.encode('UTF-8')
            postfix = hashlib.md5(bpostfix).hexdigest()
            name = self.prefix + midfix + "_-_" + postfix + extension
        return name

    @staticmethod
    def dict_to_postfix(d, ignore_pars=[]):
        d = sorted(d.items())
        first_item = d[0]
        postfix = "%s-%s"%(first_item[0], first_item[1])
        for k,v in d[1:]:
            if k not in ignore_pars:
                postfix += ",_%s-%s"%(k,v)
        return postfix

