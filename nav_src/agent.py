"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""
import json
import yaml
import re
import warnings
import numpy as np
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Dict, Union
import subprocess
import socket
import time
from env import R2RNavBatch
from argparse import Namespace
from agent_base import BaseAgent

from langchain import HuggingFacePipeline
from langchain.agents.agent import AgentExecutor, AgentAction, AgentOutputParser
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "sk-Rlk4nftf04HJqggDupZ7uM4Ur7TNUIHhlAlStDI2hCQtTLc5"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    BaseOutputParser,
    OutputParserException
)
from langchain.base_language import BaseLanguageModel

from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from prompt.planner_prompt import (
    ACTION_PROMPT,
    HISTORY_PROMPT,
    PLANNER_PROMPT,
    BACK_TRACE_PROMPT,
    MAKE_ACTION_TOOL_NAME,
    MAKE_ACTION_TOOL_DESCRIPTION,
    BACK_TRACE_TOOL_NAME,
    BACK_TRACE_TOOL_DESCRIPTION,
    VLN_ORCHESTRATOR_PROMPT,
    VLN_GPT4_PROMPT,
    VLN_GPT35_PROMPT,
    VLN_ROBOT_DOG_PROMPT,
)

FINAL_ANSWER_ACTION = "Final Answer:"
EXCEPTION_TOOL_NAME = "_Exception"
MAX_SCRATCHPAD_LENGTH = 7000

MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)

class RobotDogCommandParser(AgentOutputParser):
    """
    Parses the LLM output to extract the direct command for the robot dog.
    """
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text,
            )

        #regex = r"Command\s*:\s*(.*)"
        regex = r"(.*)Command\s*:\s*(.*)"
        match = re.search(regex, text, re.DOTALL)

        if not match:
            raise OutputParserException(f"Could not parse command from LLM output: `{text}`")
        thought = match.group(1).strip()
        action_input = match.group(2).strip()
        return AgentAction(tool="robot_command_executor", tool_input=action_input, log=text)

class NavGPTOutputParser(AgentOutputParser):
    """MRKL Output parser for the chat agent."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*\"?([a-fA-F0-9]{32})\"?"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "mrkl-NavGPT"

class VLNAgent(ZeroShotAgent):

    history: Optional[List[str]] = None 

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        nav_step = 1
        for i, (action, observation) in enumerate(intermediate_steps):
            thoughts += action.log
            if (i == len(intermediate_steps) - 1) or (action.tool != MAKE_ACTION_TOOL_NAME):
                thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
            else:
                thoughts += f"\n{self.observation_prefix}{self.history[nav_step]}\n{self.llm_prefix}"
                nav_step += 1
        return thoughts

    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)[-MAX_SCRATCHPAD_LENGTH:]
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        if "init_observation" in kwargs and self.llm_chain.prompt.template == VLN_ROBOT_DOG_PROMPT:
            # 如果是机器狗模式，从输入中移除 init_observation
            kwargs.pop("init_observation", None)
        
        elif len(intermediate_steps) > 0:
            # 这是原有的模拟器逻辑，保留它
            kwargs["init_observation"] = self.history[0]

        full_inputs = {**kwargs, **new_inputs}
        return full_inputs
        # if len(intermediate_steps) == 0:
        #     full_inputs = {**kwargs, **new_inputs}
        # else:
        #     kwargs["init_observation"] = self.history[0]
        #     full_inputs = {**kwargs, **new_inputs}
        # return full_inputs

class NavAgent(BaseAgent):
    def __init__(
            self, 
            env: R2RNavBatch, 
            config: Namespace):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env)
        self.config = config

        if config.llm_model_name.split('-')[0] == 'gpt':
            self.llm = OpenAI(
                temperature=config.temperature,
                model_name=config.llm_model_name,
            )
        elif config.llm_model_name == 'llama-2-13b':
            from LLMs.Langchain_llama import Custom_Llama
            ckpt_dir = "LLMs/llama/llama-2-13b"
            tokenizer_path = "LLMs/llama/tokenizer.model"
            self.llm = Custom_Llama.from_model_id(
                temperature=config.temperature,
                ckpt_dir = ckpt_dir,
                tokenizer_path = tokenizer_path,
                max_seq_len = 8000,
                max_gen_len = 500,
                max_batch_size = 1,
            )
        # elif config.llm_model_name == 'Vicuna-v1.5-13b':
        #     from LLMs.Langchain_Vicuna import Custom_Vicuna
        #     self.llm = Custom_Vicuna.from_config(
        #         config = config,
        #     )
        # elif config.llm_model_name == 'FlanT5XXL':
        #     from LLMs.Langchain_FlanT5 import Custom_FlanT5
        #     self.llm = Custom_FlanT5.from_config(
        #         config = config,
        #     )
        # elif config.llm_model_name == 'Emu-14B':
        #     from LLMs.Langchain_Emu import Custom_Emu
        #     self.llm = Custom_Emu.from_config(
        #         config = config,
        #     )
        # else:
        #     from LLMs.Langchain_InstructBLIP import Custom_NavGPT_InstructBLIP
        #     self.llm = Custom_NavGPT.from_config(
        #         config = config,
        #     )

        self.output_parser = NavGPTOutputParser()
        self.agent_executor = self.create_vln_agent()

        plan_prompt = PromptTemplate(
            template=PLANNER_PROMPT,
            input_variables=["instruction"],
        )
        self.plan_chain = LLMChain(llm=self.llm, prompt=plan_prompt)
    
    def parse_action(self, llm_output: str) -> Tuple[str, str]:
        regex = r"(.*?)Final Answer:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        thought = match.group(1).strip()
        action = match.group(2).strip(" ").strip('"').strip("'")

        return thought, action

    def get_his_viewpoints(self) -> str:
        '''Return the history of visited viewpoints for back tracing.'''
        his_viewpoints = ''
        # The last vp is not included in the history
        for i, detail in enumerate(self.traj[0]['details'][:-1]):
            viewpointID = detail['viewpointID']
            viewpoint_ob = detail['feature']
            his_viewpoints += f"Step {i+1}. Viewpoint ID '{viewpointID}':\n {viewpoint_ob}\n\n"
        return his_viewpoints
    
    def get_history(self, obs: dict, angle: str) -> str:
        '''Return the history of actions taken.'''
        history = f'{angle}\nCurrent viewpoint "{obs["viewpoint"]}": Scene from the viewpoint is a {obs["obs_summary"]}'
        return history

    def get_navigable_str(self, cur_heading: float, cur_elevation: float, navigable: dict) -> str:
        '''Return the navigable viewpoints as a string.'''
        navigable_str = ''

        for vp, items in navigable.items():
            heading = np.rad2deg(items['heading'])
            elevation = np.rad2deg(items['elevation'])
            distance = items['distance']
            rel_heading = heading - cur_heading
            rel_elevation = elevation - cur_elevation

            if self.config.use_relative_angle:
                navigable_str += f"'{vp}':\nheading: {rel_heading:.2f}, elevation: {rel_elevation:.2f}, distance: {distance:.2f}\n"
            else:
                navigable_str += f"'{vp}':\nheading: {heading:.2f}, elevation: {elevation:.2f}, distance: {distance:.2f}\n"

        return navigable_str

    def modify_heading_angles(self, heading_angle, observation_list, candidate_dict, object_list):
        # Function to normalize an angle to the range of -180 to 180
        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle <= -180:
                angle += 360
            return angle
        
        def angle_to_left_right(angle):
            return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"
        
        # Define the directions
        # directions = ['Front', 'Front Right', 'Right', 'Rear Right', 'Rear', 'Rear Left', 'Left', 'Front Left']
        directions = ['Front', 'Front Right', 'Front Left']

        # Calculate the range of heading angles belonging to each direction
        range_idx = int((heading_angle - 22.5) // 45) + 1
        # obs_idx = [(i + range_idx) % 8 for i in range(8)]
        obs_idx = [(i + range_idx) % 3 for i in range(3)]
        
        # Initialize a dictionary to store the candidate viewpoints for each direction
        candidate_range = {}
        if not self.config.use_navigable:
            for viewpoint_id, viewpoint_data in candidate_dict.items():
                viewpoint_heading = np.rad2deg(viewpoint_data['heading'])
                vp_range_idx = int((viewpoint_heading - 22.5) // 45) + 1
                rel_viewpoint_heading = viewpoint_heading - heading_angle
                rel_viewpoint_heading = normalize_angle(rel_viewpoint_heading)
                rel_viewpoint_heading = angle_to_left_right(rel_viewpoint_heading)
                vp_description = rel_viewpoint_heading + f', {viewpoint_data["distance"]:.2f}m'
                # rel_range_idx = (vp_range_idx - range_idx) % 8
                candidate_range.setdefault(vp_range_idx, {}).update({viewpoint_id: vp_description})

        # Calculate the relative angle ranges based on the heading angle
        angle_ranges = [(angle - 22.5 - heading_angle, angle + 22.5 - heading_angle) for angle in range(0, 360, 45)]
        
        # Initialize an empty list to store the formatted strings
        formatted_strings = []
        
        # Iterate through the directions, angle ranges, and observation strings
        for direction, idx in zip(directions, obs_idx):
            # Calculate the relative angles and normalize them
            rel_angle1 = normalize_angle(angle_ranges[idx][0])
            rel_angle2 = normalize_angle(angle_ranges[idx][1])

            # Convert the angles to "left n" or "right n"
            left_right1 = angle_to_left_right(rel_angle1)
            left_right2 = angle_to_left_right(rel_angle2)
            
            # Create the formatted string
            formatted_string = f"{direction}, range ({left_right1} to {left_right2}): \n'{observation_list[idx]}'"

            # Add the objects to the formatted string
            object_dict = {}
            if len(object_list[idx]) > 0:
                object = object_list[idx]
                for obj, obj_data in object.items():
                    rel_obj_heading = obj_data['heading'] - heading_angle
                    rel_obj_heading = normalize_angle(rel_obj_heading)
                    rel_obj_heading = angle_to_left_right(rel_obj_heading)
                    object_dict[obj] = f'{rel_obj_heading}, {obj_data["distance"]:.2f}m'
                formatted_string += f'\n{direction} Objects in 3m: {object_dict}'
            else:
                formatted_string += f'\n{direction} Objects in 3m: None'

            # Add the candidate viewpoints to the formatted string
            if candidate_range.get(idx):
                formatted_string += f'\n{direction} Navigable Viewpoints:{candidate_range[idx]}'
            else:
                formatted_string += f'\n{direction} Navigable Viewpoints: None'
   
            # Add the formatted string to the list
            formatted_strings.append(formatted_string)
        
        # Join the formatted strings into a single output string
        output_string = '\n'.join(formatted_strings)
        
        return output_string

    def init_trajecotry(self, obs: List[dict]):
        """Initialize the trajectory with the given observation."""
        # Record the navigation path
        self.traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': [],
        } for ob in obs]
        if self.config.agent_mode == 'robot_dog':
            # 在机器狗模式下，历史记录不应包含任何初始观察。
            # 提示词会强制它先扫描。
            self.agent_executor.agent.history = ['Navigation start, no actions taken yet.']
        else:
            # 模拟器模式：保持原有逻辑
            self.agent_executor.agent.history = [f'Navigation start, no actions taken yet.\nCurrent viewpoint "{obs[0]["viewpoint"]}": Scene from the viewpoint is a {obs[0]["obs_summary"]}']

        # Record the history of actions taken
        #self.agent_executor.agent.history = [f'Navigation start, no actions taken yet.\nCurrent viewpoint "{obs[0]["viewpoint"]}": Scene from the viewpoint is a {obs[0]["obs_summary"]}']

    def _create_make_action_tool(
            self,
            llm: BaseLanguageModel,
    ) -> Tool:
        """Create a tool to make single action prediction in MP3D.

        The tool is invoked with the simulation environment and records the
        action taken by the agent.
        The tool interacts with the environment to obtain the current observation, 
        uses the LLM to predict the next action, and to summarize the previous trajectory
        into history.
        """

        action_prompt = PromptTemplate(
            template=ACTION_PROMPT,
            input_variables=["action_plan", "observation", "history", "navigable_viewpoints"],
        )
        history_prompt = PromptTemplate(
            template=HISTORY_PROMPT,
            input_variables=["history", "previous_action", "observation"],
        )
        self.action_chain = LLMChain(llm=llm, prompt=action_prompt)
        self.history_chain = LLMChain(llm=llm, prompt=history_prompt)

        def _make_action(*args, **kwargs) -> str:
            '''Make single step action in MatterSim.'''
            # Get current observation
            cur_obs = self.env._get_obs()[0]

            # Get current feature
            feature = cur_obs['obs']
            heading = np.rad2deg(cur_obs['heading'])
            elevation = np.rad2deg(cur_obs['elevation'])
            objects = cur_obs['objects']
            orientation = f'\nheading: {heading:.2f}, elevation: {elevation:.2f}'
            navigable = cur_obs['candidate']
            if self.config.use_relative_angle:
                feature = self.modify_heading_angles(heading, feature, navigable, objects)
            if self.config.use_navigable:
                navigable = self.get_navigable_str(heading, elevation, navigable)

            if self.config.use_tool_chain:
                # Get current action plan
                action_plan = self.cur_action_plan
                # Single step action
                LLM_action_output = self.action_chain.run(
                    action_plan = action_plan, 
                    observation = feature, 
                    history = self.agent_executor.agent.history[-1], 
                    navigable_viewpoints = navigable
                )
                # Parse LLM output, action is the next viewpoint ID
                thought, action = self.parse_action(LLM_action_output)
            else:
                action = args[0].strip(" ").strip('"').strip("'")

            # Make the action in Simulator
            if action not in self.env.env.sims[0].navigable_dict.keys():
                # Update history
                history = f'ViewpointID "{action}" is not valid, no action taken for the agent.'
                self.agent_executor.agent.history.append(history)
                if self.config.use_navigable:
                    return f"\nViewpointID '{action}' is not valid, agent not moved. DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose from current viewpoints are: {[key for key in navigable.keys()]}.\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                else:
                    return f"\nViewpointID '{action}' is not valid, agent not moved. DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose from current viewpoints are: {[key for key in navigable.keys()]}.\n\tCurrent Viewpoint:\n{feature}"
            else:
                turned_angle, new_obs = self.make_equiv_action([action])

            # Update the current feature
            new_feature = new_obs['obs']
            new_feature_sum = new_obs['obs_summary']
            new_navigable = new_obs['candidate']
            new_objects = new_obs['objects']
            new_heading = np.rad2deg(new_obs['heading'])
            new_elevation = np.rad2deg(new_obs['elevation'])
            if self.config.use_relative_angle:
                new_feature = self.modify_heading_angles(new_heading, new_feature, new_navigable, new_objects)
            new_orientation = f'\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}'
            if self.config.use_navigable:
                new_navigable = self.get_navigable_str(new_heading, new_elevation, new_navigable)

            # Update history
            if self.config.use_history_chain:
                history = self.history_chain.run(
                    observation = new_feature_sum, 
                    history = self.agent_executor.agent.history[-1], 
                    previous_action = turned_angle
                )
            else:
                history = self.get_history(new_obs, turned_angle)
            
            self.agent_executor.agent.history.append(history)
            # Record single step detail
            if self.config.use_tool_chain:
                detail = {
                    "viewpointID": action,
                    "turned_angle": turned_angle,
                    "acion_maker_thought": thought,
                    "feature": new_feature,
                    "history": self.agent_executor.agent.history[-1],
                }
            else:
                detail = {
                    "viewpointID": action,
                    "turned_angle": turned_angle,
                    "feature": new_feature,
                    "history": self.agent_executor.agent.history[-1],
                }
            self.traj[0]['details'].append(detail)
            # Return LLM chain output as the observation of tool
            if self.config.use_tool_chain:
                return f"\n\tAction_maker Thought:\n{thought}\n\tAction_maker Action:\n{turned_angle}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
            elif self.config.use_relative_angle:
                if self.config.use_navigable:
                    return f"\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f'\nCurrent Viewpoint "{action}":\n{new_feature}'
            else:
                if self.config.use_navigable:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}"
                

        return Tool(
            name=MAKE_ACTION_TOOL_NAME,
            func=_make_action,
            description=MAKE_ACTION_TOOL_DESCRIPTION,
        )

    def _create_back_trace_tool(
            self,
            llm: BaseLanguageModel,
    ) -> Tool:
        """Create a tool to back trace during navigation.

        The tool is invoked with the history of navigation trajectory.
        Using the LLM to find a viewpoint on the trajectory to back trace to.
        """
        prompt = PromptTemplate(
            template=BACK_TRACE_PROMPT,
            input_variables=["action_plan", "history", "observation"],
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        def _back_trace(*args, **kwargs) -> str:
            '''Back trace the action plan.'''
            cur_obs = self.env._get_obs()[0]

            # Get current feature
            feature = cur_obs['obs']
            navigable = cur_obs['candidate']
            objects = cur_obs['objects']
            heading = np.rad2deg(cur_obs['heading'])
            elevation = np.rad2deg(cur_obs['elevation'])
            orientation = f'\nheading: {heading:.2f}, elevation: {elevation:.2f}'
            if self.config.use_relative_angle:
                feature = self.modify_heading_angles(heading, feature, navigable, objects)
            if self.config.use_navigable:
                navigable = self.get_navigable_str(heading, elevation, navigable)

            if self.config.use_tool_chain:
                # Get current action plan
                action_plan = self.cur_action_plan
                # Get all previous viewpoints observation
                previous_vp = self.get_his_viewpoints()
                # Back trace
                LLM_output = chain.run(action_plan = action_plan, observation = previous_vp, history = self.agent_executor.agent.history[-1])
                # Parse LLM output, action is the next viewpoint ID
                thought, action = self.parse_action(LLM_output)
            else:
                action = args[0].strip(" ").strip('"').strip("'")

            # Make the action in Simulator
            if action not in self.env.env.sims[0].navigable_dict.keys():
                if self.config.use_navigable:
                    return f"\nViewpointID '{action}' is not valid. DO NOT fabricate nonexistent IDs.\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                else:
                    return f"\nViewpointID '{action}' is not valid. DO NOT fabricate nonexistent IDs.\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"
            else:
                _, new_obs = self.make_equiv_action([action])
            
            # Update the current feature
            new_feature = new_obs['obs']
            new_navigable = new_obs['candidate']
            new_objects = new_obs['objects']
            new_heading = np.rad2deg(new_obs['heading'])
            new_elevation = np.rad2deg(new_obs['elevation'])
            new_orientation = f'\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}'
            if self.config.use_relative_angle:
                new_feature = self.modify_heading_angles(new_heading, new_feature, new_navigable, new_objects)
            if self.config.use_navigable:
                new_navigable = self.get_navigable_str(new_heading, new_elevation, new_navigable)

            # Update history
            history = self.get_history(new_obs, 'Seems going in a wrong way, back trace to a previous point.')
            self.agent_executor.agent.history.append(history)
            # Record single step detail
            if self.config.use_tool_chain:
                return f"\tBack_tracer Thought:\n{thought}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
            elif self.config.use_relative_angle:
                if self.config.use_navigable:
                    return f"\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\nCurrent Viewpoint:{action}\n{new_feature}"
            else:
                if self.config.use_navigable:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}"

        return Tool(
            name=BACK_TRACE_TOOL_NAME,
            func=_back_trace,
            description=BACK_TRACE_TOOL_DESCRIPTION,
        )

    def create_vln_agent(
        self,
    ) -> AgentExecutor:
        """Instantiate API planner and controller for a given trajectory.

        We use a top-level "orchestrator" agent to invoke the planner and controller,
        rather than a top-level planner
        that invokes a controller with its plan. This is to keep the planner simple.
        """

        # --- ADD THIS DEBUG LINE ---
        print(f"\n[DEBUG] Creating agent in mode: '{self.config.agent_mode}'\n")
        # --- END DEBUG LINE ---
        self.action_maker = self._create_make_action_tool(self.llm)
        self.back_tracer = self._create_back_trace_tool(self.llm)

        if self.config.agent_mode == 'robot_dog':
            # --- Logic for Robot Dog ---
            def _execute_robot_command(command: str) -> str:
                """
                This function takes the natural language command from the LLM,
                sends it to the robot's control API, and returns the robot's feedback.
                """
                # print(f"--- Sending command to robot: {command} ---")
                # # ===============================================================
                # # TODO: IMPLEMENT YOUR ROBOT'S API CALLS HERE
                # # For example: feedback = my_robot_api.send_command(command)
                # # ===============================================================
                # feedback = f"Command '{command}' executed. New observation: I see the wall of the office in front of me."
                # print(f"--- Received feedback from robot: {feedback} ---")
                # return feedback

                ROBOT_IP = "192.168.8.2"
                ROBOT_PORT = 12345
                
                response_from_robot = ""
                blockage_info = ""  # 新增变量，用于存储障碍物信息
                print(f"--- Received command from LLM: {command} ---")
                
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(10) # 设置10秒连接超时
                        s.connect((ROBOT_IP, ROBOT_PORT))
                        s.sendall(command.encode('utf-8'))
                            # 可选：等待服务器的回执
                        response_from_robot = s.recv(1024).decode('utf-8')
                        print(f"--- Received from robot: {response_from_robot} ---")
                        if "failed" in response_from_robot.lower() or "blocked" in response_from_robot.lower():
                            blockage_info = response_from_robot
                except socket.timeout:
                    error_msg = f"Connection to robot at {ROBOT_IP}:{ROBOT_PORT} timed out."
                    print(f"--- {error_msg} ---")
                    # 如果连接失败，直接返回错误，不执行后续的图像处理
                except Exception as e:
                    error_msg = f"Failed to send command to robot: {e}"
                    print(f"--- {error_msg} ---")
                    # 如果连接失败，直接返回错误
                # if "failed" in response_from_robot.lower() or "blocked" in response_from_robot.lower():
                #     # 如果路径被阻挡，直接将这个信息作为观察结果返回给LLM
                #     return response_from_robot
                # 在发送运动指令后，可以稍作等待，让机器人有时间执行动作
                # time.sleep(10.0) # 等待2秒，可以根据动作调整
                print(f"--- Now getting observation from environment... ---")
                
                # 2. 定义需要执行的命令
                # 命令1：运行 ROS2 节点以捕获并保存图片
                # 注意：这需要在一个已经 source 了 ROS2 环境的 shell 中运行
                capture_command = "/usr/bin/python3 /home/xen/ros2_ws/src/realsense-ros/realsense2_camera/scripts/image_processor_node.py"
                
                # 命令2：运行 VLM 脚本来处理图片并生成描述
                # 假设 blip2.py 所在的 conda 环境是 navgpt
                vlm_command = "/home/xen/anaconda3/envs/NavGPT/bin/python /home/xen/vlm/blip2.py"

                try:
                    # 3. 执行图片捕获命令
                    print("--- Running image capture node... ---")
                    # 使用 shell=True 来处理复杂的命令链 (&&)
                    # 设置一个超时以防节点卡住
                    capture_result = subprocess.run(
                        capture_command, 
                        shell=True, 
                        capture_output=True, 
                        text=True, 
                        timeout=15,
                        executable='/bin/bash' # 明确使用 bash
                    )
                    
                    if capture_result.returncode != 0:
                        error_message = f"Image capture failed with error: {capture_result.stderr}"
                        print(f"--- {error_message} ---")
                        return error_message
                    
                    print("--- Image capture successful. ---")

                    # 4. 执行 VLM 描述生成命令
                    print("--- Running VLM to get scene description... ---")
                    vlm_result = subprocess.run(
                        vlm_command, 
                        shell=True, 
                        capture_output=True, 
                        text=True, 
                        timeout=60 # VLM 可能需要更长的时间
                    )

                    if vlm_result.returncode != 0:
                        error_message = f"VLM script failed with error: {vlm_result.stderr}"
                        print(f"--- {error_message} ---")
                        return error_message

                    # 5. 提取 VLM 的输出作为 feedback
                    feedback = vlm_result.stdout.strip()
                    if not feedback:
                        feedback = "VLM generated an empty description."

                except subprocess.TimeoutExpired:
                    timeout_msg = "A subprocess timed out."
                    print(f"--- {timeout_msg} ---")
                    return timeout_msg
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {e}"
                    print(f"--- {error_msg} ---")
                    return error_msg
                if blockage_info:
                    # 如果存在障碍物信息，将其附加到VLM反馈的后面
                    final_feedback = f"{feedback}. {blockage_info}"
                else:
                    final_feedback = feedback
                print(f"--- Received feedback from VLM: {final_feedback} ---")
                return final_feedback


            robot_command_tool = Tool(
                name="robot_command_executor",
                func=_execute_robot_command,
                description="Executes a natural language command on the robot and returns the result."
            )
            tools = [robot_command_tool]
            prompt = PromptTemplate(
                template=VLN_ROBOT_DOG_PROMPT, # Use the new prompt
                input_variables=["action_plan", "agent_scratchpad"],
            )
            output_parser = RobotDogCommandParser()

        else:
            # --- Original Logic for Simulator ---
            if self.config.use_single_action:
                tools = [self.action_maker]
                prompt = PromptTemplate(
                    template=VLN_GPT4_PROMPT if self.config.llm_model_name == 'gpt-4' else VLN_GPT35_PROMPT,
                    input_variables=["action_plan", "init_observation", "agent_scratchpad"],
                    partial_variables={
                        "tool_names": ", ".join([tool.name for tool in tools]),
                        "tool_descriptions": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                    },
                )
            else: # Fallback to tool_chain or other modes
                tools = [self.action_maker, self.back_tracer]
                prompt = PromptTemplate(
                    template=VLN_ORCHESTRATOR_PROMPT,
                    input_variables=["action_plan", "init_observation", "agent_scratchpad"],
                    partial_variables={
                        "tool_names": ", ".join([tool.name for tool in tools]),
                        "tool_descriptions": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                    },
                )
            output_parser = self.output_parser # Use the original parser

        agent = VLNAgent(
            llm_chain=LLMChain(llm=self.llm, prompt=prompt),
            allowed_tools=[tool.name for tool in tools],
            output_parser=output_parser # Use the dynamically selected parser
        )
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            max_iterations=self.config.max_iterations,
        )
        # tools = [
        #     self.action_maker,
        #     self.back_tracer
        # ]

        # if self.config.use_tool_chain:
        #     prompt = PromptTemplate(
        #         template=VLN_ORCHESTRATOR_PROMPT,
        #         input_variables=["action_plan", "init_observation", "observation", "agent_scratchpad"],
        #         partial_variables={
        #             "tool_names": ", ".join([tool.name for tool in tools]),
        #             "tool_descriptions": "\n".join(
        #                 [f"{tool.name}: {tool.description}" for tool in tools]
        #             ),
        #         },
        #     )
        # elif self.config.use_single_action:
        #     tools = [self.action_maker]
        #     prompt = PromptTemplate(
        #         template=VLN_GPT4_PROMPT if self.config.llm_model_name == 'gpt-4' else VLN_GPT35_PROMPT,
        #         input_variables=["action_plan", "init_observation", "agent_scratchpad"],
        #         partial_variables={
        #             "tool_names": ", ".join([tool.name for tool in tools]),
        #             "tool_descriptions": "\n".join(
        #                 [f"{tool.name}: {tool.description}" for tool in tools]
        #             ),
        #         },
        #     )
        # else:
        #     prompt = PromptTemplate(
        #         template=VLN_ORCHESTRATOR_PROMPT,
        #         input_variables=["action_plan", "init_observation", "agent_scratchpad"],
        #         partial_variables={
        #             "tool_names": ", ".join([tool.name for tool in tools]),
        #             "tool_descriptions": "\n".join(
        #                 [f"{tool.name}: {tool.description}" for tool in tools]
        #             ),
        #         },
        #     )
        # agent = VLNAgent(
        #     llm_chain=LLMChain(llm=self.llm, prompt=prompt),
        #     allowed_tools=[tool.name for tool in tools],
        #     output_parser = self.output_parser
        # )
        # return AgentExecutor.from_agent_and_tools(
        #     agent=agent, 
        #     tools=tools, 
        #     verbose=True, 
        #     handle_parsing_errors = True,
        #     return_intermediate_steps=True,
        #     max_iterations=self.config.max_iterations,
        # )
    
    def make_equiv_action(self, actions: List[str]) -> str:
        """
        Interface between Panoramic view and Egocentric view
        Take in the next viewpoint ID and move the agent to that viewpoint
        return the turned angle and new observation
        """
        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle <= -180:
                angle += 360
            return angle

        def angle_to_left_right(angle):
            return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"
        
        # Get current agent facing angle
        cur_obs = self.env._get_obs()[0]
        cur_heading = np.rad2deg(cur_obs['heading'])
        # Make the action
        new_obs = self.env.step(actions)[0]
        new_heading = np.rad2deg(new_obs['heading'])
        # Record the trajectory
        self.traj[0]['path'].append(self.env.env.sims[0].gmap.bfs_shortest_path(cur_obs['viewpoint'], actions[0])[1:])
        # Calculate the turned angle
        turned_angle = new_heading - cur_heading
        # Generate action description
        cur_heading = angle_to_left_right(normalize_angle(cur_heading))
        new_heading = angle_to_left_right(normalize_angle(new_heading))
        action_description = f'Turn heading direction {turned_angle:.2f} degrees from {cur_heading} to {new_heading}.'
        return action_description, new_obs

    def rollout(self, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        # Initialize the trajectory
        self.init_trajecotry(obs)

        # Load the instruction
        instructions = [ob['instruction'] for ob in obs]
        if self.config.load_instruction:
            action_plans = instructions
        elif self.config.load_action_plan:
            action_plans = [ob['action_plan'] for ob in obs]
        else:
            action_plans = []
            for instruction in instructions:
                action_plan = self.plan_chain.run(instruction = instruction)
                action_plans.append(action_plan)

        for i, init_ob in enumerate(obs):
            self.cur_action_plan = action_plans[i]

                       # Take the first action
            if self.config.agent_mode == 'robot_dog':
                print("\n" + "="*60)
                print(f"STARTING NEW ROBOT DOG TRAJECTORY")
                print(f"INSTRUCTION: {self.cur_action_plan}")
                #print(f"INITIAL OBSERVATION: {init_observation}")
                print("="*60 + "\n")
                input = {
                    'action_plan': self.cur_action_plan,
                    
                }
            elif self.config.use_tool_chain:
                first_obs = self.action_maker('')
                input = {
                    'action_plan': self.cur_action_plan,
                    'init_observation': init_ob['obs_summary'],
                    'observation': first_obs,
                }
            else:
                # Get current feature
                feature = init_ob['obs']
                navigable = init_ob['candidate']
                objects = init_ob['objects']
                heading = np.rad2deg(init_ob['heading'])
                elevation = np.rad2deg(init_ob['elevation'])
                orientation = f'\nheading: {heading:.2f}, elevation: {elevation:.2f}'
                if self.config.use_relative_angle:
                    feature = self.modify_heading_angles(heading, feature, navigable, objects)
                if self.config.use_navigable:
                    navigable = self.get_navigable_str(heading, elevation, navigable)

                if self.config.use_relative_angle:
                    if self.config.use_navigable:
                        init_observation = f"\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                    else:
                        init_observation = f"\n\tCurrent Viewpoint:\n{feature}"
                else:
                    if self.config.use_navigable:
                        init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                    else:
                        init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"

                input = {
                    'action_plan': self.cur_action_plan,
                    'init_observation': init_observation,
                }
            output = self.agent_executor(input)

            self.traj[i]['llm_output'] = output['output']
            self.traj[i]['action_plan'] = output['action_plan']
            # extract agent's thought from llm output
            intermediate_steps = output['intermediate_steps']
            self.traj[i]['llm_thought'] = []
            self.traj[i]['llm_observation'] = []
            for action, observation in intermediate_steps:
                thought = action.log
                self.traj[i]['llm_thought'].append(thought)
                self.traj[i]['llm_observation'].append(observation)

        return self.traj