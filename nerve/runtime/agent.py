import pathlib

import click
from loguru import logger

import nerve.runtime.state as state
from nerve.generation import Engine, Usage, WindowStrategy
from nerve.generation.conversation import FullHistoryStrategy
from nerve.generation.litellm import LiteLLMEngine
from nerve.models import Configuration, Tool
from nerve.runtime import Runtime


class Agent:
    def __init__(
        self,
        runtime: Runtime,
        configuration: Configuration,
        generation_engine: Engine,
        conv_window_strategy: WindowStrategy,
    ):
        # runtime data
        self.runtime = runtime
        # agent definition / configuration
        self.configuration = configuration
        # generation engine
        self.generation_engine = generation_engine
        # conversation window strategy
        self.conv_window_strategy = conv_window_strategy

        state.on_event("agent_created", {"agent": self})

    @classmethod
    def create(
        cls,
        generator: str,
        configuration: Configuration,
        start_state: dict[str, str] | None = None,
        window_strategy: WindowStrategy = FullHistoryStrategy(),
        working_dir: pathlib.Path = pathlib.Path.cwd(),
        name: str = "agent",
    ) -> "Agent":
        """
        Create an agent from a generator and configuration.

        Args:
            generator: The generator string to use.
            configuration: The configuration to use.
            start_state: Initial variables for the agent.
            window_strategy: How to handle conversation history.
            working_dir: The working directory to use.
            name: The name of the agent.
        """

        if start_state:
            state.update_variables(start_state)

        # import tools.py by default if found
        tools_py_path = working_dir / "tools.py"
        if tools_py_path.exists():
            configuration.tools.append(
                Tool(
                    path=str(tools_py_path),
                )
            )

        runtime = Runtime.build(
            working_dir=working_dir,
            name=name,
            generator=generator,
            using=configuration.using,
            jail=configuration.jail,
            tools=configuration.tools,
        )

        return cls(
            runtime=runtime,
            configuration=configuration,
            generation_engine=LiteLLMEngine(generator, window_strategy, runtime.tools),
            conv_window_strategy=window_strategy,
        )

    @classmethod
    def create_from_file(
        cls,
        generator: str,
        config_file_path: pathlib.Path,
        window_strategy: WindowStrategy = FullHistoryStrategy(),
        start_state: dict[str, str] | None = None,
    ) -> "Agent":
        config = Configuration.from_path(config_file_path)
        if config.is_legacy:
            logger.error(f"legacy format detected, update to the 1.0.0 format: {config_file_path}")
            exit(1)

        stem = config_file_path.stem
        working_dir = (config_file_path if config_file_path.is_dir() else config_file_path.parent).absolute()
        return cls.create(
            generator,
            config,
            start_state,
            window_strategy,
            working_dir,
            stem if stem not in ("task", "agent") else working_dir.stem,
        )

    def _get_system_prompt(self) -> str | None:
        if not self.configuration.agent:
            return None

        raw = self.configuration.agent

        for name, value in state.get_knowledge().items():
            raw += f"\n\n## {name.capitalize()}\n\n{value}"

        return state.interpolate(raw)

    def _get_prompt(self) -> str:
        if not self.configuration.task:
            self.configuration.task = state.on_user_input_needed("task", "Describe the task: ")

        return state.interpolate(self.configuration.task)

    async def step(self) -> Usage:
        logger.debug(f"agent {self.runtime.name} step")

        if state.get_current_actor() != self:
            state.on_task_started(self)

        try:
            system_prompt = self._get_system_prompt()
            prompt = self._get_prompt()
            extra_tools = state.get_extra_tools()
            logger.debug(f"system_prompt: {system_prompt}")
            logger.debug(f"prompt: {prompt}")
            logger.debug(f"extra_tools: {extra_tools}")

            state.on_event(
                "agent_step",
                {
                    "agent_name": self.runtime.name,
                    "generator": self.runtime.generator,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                },
            )

            usage = await self.generation_engine.step(system_prompt, prompt, extra_tools)
            logger.debug(f"usage: {usage}")
            return usage
        except click.exceptions.MissingParameter as e:
            logger.error(e)
            exit(1)
        except Exception as e:
            state.on_event(
                "error",
                {
                    "agent_name": self.runtime.name,
                    "error": e,
                },
            )
            import traceback

            logger.error(f"Exception during agent step: {e}")
            logger.error(traceback.format_exc())
            quit()

    async def run(
        self,
        max_steps: int = 500,
        timeout: int | None = None,
        start_state: dict[str, str] | None = None,
    ) -> None:
        """
        Shortcut method to run a single agent without explicitly creating a Flow object.

        Runs the agent until one of the following conditions is met:

        - The task is completed.
        - The maximum number of steps is reached.
        - The timeout is reached.

        Args:
            max_steps: The maximum number of steps to run.
            timeout: The timeout in seconds.
            start_state: Initial variables for the agent.
        """

        # import here to avoid circular import
        from nerve.runtime.flow import Flow

        await Flow.build(
            actors=[self],
            max_steps=max_steps,
            timeout=timeout,
            start_state=start_state,
        ).run()
