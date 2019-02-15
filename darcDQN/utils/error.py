
class Error(Exception):
    pass


class UnknownArgument(Error):
    """Raised when the user enters an unknown argument.
    """
    pass


class UnregisteredAgent(Error):
    """Raised when the user request an agent from the registry that does not exist.
    """
    pass


class UnknownAgentVersion(Error):
    """Raised when the user requests a version of an agent that does not exist.
    """
    pass
