from setuptools import setup, Extension

module = Extension('i8',
                    sources = ['i8module.c', 'i8_ops.c'],
                    include_dirs=['.'])

setup(name = 'i8',
      version = '0.1',
      description = 'This is a demo package for i8 arrays',
      ext_modules = [module])