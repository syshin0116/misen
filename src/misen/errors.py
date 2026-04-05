"""misen exception hierarchy."""


class MisenError(Exception):
    """Base exception for all misen errors."""


class BlockError(MisenError):
    """A block failed during execution."""


class MergeConflictError(MisenError):
    """Parallel blocks produced conflicting output keys."""


class LoopMaxIterationsError(MisenError):
    """Loop exceeded the maximum number of iterations."""


class RegistryKeyError(MisenError, KeyError):
    """Block not found in the registry."""
