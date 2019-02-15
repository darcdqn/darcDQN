import re
from gym import logger
from darcDQN.utils import error

"""Registry from OpenAI gym environments"""

agent_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$')


def load(name):
    import pkg_resources  # takes ~400ms to load, so we import it lazily
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(name))
    result = entry_point.resolve()
    return result


class AgentSpec(object):
    """A specification for a particular instance of an agent. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official agent ID

        entry_point (str): The Python entrypoint of the agent class
                                     (e.g. module.name:Class)

        kwargs (dict): The kwargs to pass to the agent class

    Attributes:
        id (str): The official environment ID
        entry_point (str): The Python entrypoint of the agent class
    """

    def __init__(self,
                 id,
                 entry_point,
                 kwargs=None):
        self.id = id

        # We may make some of these other parameters public if they're
        # useful.
        match = agent_id_re.search(id)
        if not match:
            raise error.Error(('Attempted to register malformed agent ID: {}. '
                               + '(Currently all IDs must be of the form {}.)')
                              .format(id, agent_id_re.pattern))
        self._agent_name = match.group(1)
        self._entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the agent with appropriate kwargs"""
        if self._entry_point is None:
            raise error.Error(('Attempting to make deprecated agent {}. '
                               + '(HINT: is there a newer registered version '
                               + ' of this agent?)')
                              .format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self._entry_point):
            agent = self._entry_point(**_kwargs)
        else:
            n_inputs = _kwargs['n_inputs']
            n_outputs = _kwargs['n_outputs']
            cls = load(self._entry_point)
            agent = cls(n_inputs, n_outputs)

        return agent

    def __repr__(self):
        return "AgentSpec({})".format(self.id)


class AgentRegistry(object):
    """Register an agent by ID. AgentRegistry is basically the same as
    EnvRegistry from OpenAI gym.
    """

    def __init__(self):
        self.agent_specs = {}

    def make(self, id, **kwargs):
        if len(kwargs) > 0:
            logger.info('Making new agent: %s (%s)', id, kwargs)
        else:
            logger.info('Making new agent: %s', id)
        spec = self.spec(id)
        agent = spec.make(**kwargs)
        return agent

    def all(self):
        return self.agent_specs.values()

    def spec(self, id):
        match = agent_id_re.search(id)
        if not match:
            m = (('Attempted to look up malformed agent ID: {}. '
                  + '(Currently all IDs must be of the form {}.)')
                 .format(id.encode('utf-8'), agent_id_re.pattern))
            raise error.Error(m)

        try:
            return self.agent_specs[id]
        except KeyError:
            # Parse the agent name and check to see if it matches the
            # non-version part of a valid agent
            # (could also check the exact number here)
            agent_name = match.group(1)
            matching_agents = \
                [valid_agent_name for valid_agent_name, valid_agent_spec
                 in self.agent_specs.items()
                 if agent_name == valid_agent_spec._agent_name]
            if matching_agents:
                m = 'Agent {} not found (valid versions include {})' \
                    .format(id, matching_agents)
                raise error.UnknownAgentVersion(m)
            else:
                m = 'No registered agent with id: {}'.format(id)
                raise error.UnregisteredAgent(m)

    def register(self, id, **kwargs):
        if id in self.agent_specs:
            raise error.Error('Cannot re-register id: {}'.format(id))
        self.agent_specs[id] = AgentSpec(id, **kwargs)


# Have a global registry
registry = AgentRegistry()


def register(id, **kwargs):
    return registry.register(id, **kwargs)


def make(id, **kwargs):
    return registry.make(id, **kwargs)


def spec(id):
    return registry.spec(id)

