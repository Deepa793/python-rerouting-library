class ReroutingLibraryError(Exception):
    """Base exception for the routing library."""


class BackendError(ReroutingLibraryError):
    """Base exception for backend failures."""


class LocalBackendError(BackendError):
    """Local Llama backend failed."""


class CloudBackendError(BackendError):
    """Cloud backend failed."""


class DispatchError(ReroutingLibraryError):
    """Dispatcher could not complete the request."""