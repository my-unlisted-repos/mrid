# Python 3.12 fix
in HD_BET/run.py remove:

```py
import imp
```

add the following:

```py
import importlib.machinery, importlib.util
def load_source(modname, filename):
    """Official replacement for imp.load_source from https://docs.python.org/3/whatsnew/3.12.html#imp"""
    loader = importlib.machinery.SourceFileLoader(modname, filename)
    spec = importlib.util.spec_from_file_location(modname, filename, loader=loader)
    module = importlib.util.module_from_spec(spec)
    # The module is always executed and not cached in sys.modules.
    # Uncomment the following line to cache the module.
    # sys.modules[module.__name__] = module
    loader.exec_module(module)
    return module
```

and then replace `imp.load_source` with `load_source` (I believe there is 1 occurence in the entire file).

# FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\USERNAME\\hd-bet_params\\0.model' fix.
When you get this error, download following files:

- https://zenodo.org/record/2540695/files/0.model?download=1.
- https://zenodo.org/record/2540695/files/1.model?download=1.
- https://zenodo.org/record/2540695/files/2.model?download=1.
- https://zenodo.org/record/2540695/files/3.model?download=1.
- https://zenodo.org/record/2540695/files/4.model?download=1.

The downloaded files will be called `0.model`, `1.model`, `2.model`, `3.model`, `4.model`, each is about 62 MB. Put them anywhere where they will be stored.

Then go to HD_BET/utils.py and change the following function:

```py
def get_params_fname(fold):
    return os.path.join(folder_with_parameter_files, "%d.model" % fold)
```

to wherever you put your `*.model` files, and make sure it replaces the number with `fold`, I put them into HD_BET folder so for me it is:
```py
def get_params_fname(fold):
    return rf"F:\Stuff\Programming\libs\HD-BET\HD-BET\HD_BET\{fold}.model"
```

