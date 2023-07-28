
g = "xxxx"
def inlinef():
    def inner():
        def iinner():
            return g
        return iinner
    return inner
