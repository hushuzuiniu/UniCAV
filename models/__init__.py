__all__ = ["ESMCModel", "Antiberta2Model"]


def __getattr__(name):
    if name == "ESMCModel":
        from .esmc_model import ESMCModel

        return ESMCModel
    if name == "Antiberta2Model":
        from .antiberta2_model import Antiberta2Model

        return Antiberta2Model
    raise AttributeError(name)
