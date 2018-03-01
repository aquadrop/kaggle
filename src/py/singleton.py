"""
-------------------------------------------------
   File Name：     singleton
   Description :
   Author :       deep
   date：          18-1-29
-------------------------------------------------
   Change Activity:
                   18-1-29:
                   
   __author__ = 'deep'
-------------------------------------------------
"""

class Singleton(type):
    """
    reference: https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons

    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

