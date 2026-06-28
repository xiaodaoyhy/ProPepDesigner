from typing import Union

from rdkit import RDLogger

from pepinvent.reinforcement.configuration.reinforcement_learning_configuration import ReinforcementLearningConfiguration
from pepinvent.reinforcement.learning_scenario import LearningScenario
from pepinvent.sampling.sampling import Sampling
from pepinvent.sampling.sampling_config import SamplingConfig
from pepinvent.supervised_learning.trainer.architecturedto import ArchitectureConfig
from pepinvent.supervised_learning.trainer.transformer_trainer_dist import TransformerTrainer
# from pepinvent.supervised_learning.trainer.transformer_trainer_finetune import TransformerTrainer
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model import GenerativeModel

RDLogger.DisableLog('rdApp.*')


class Manager:
    def __init__(self, configuration: Union[ReinforcementLearningConfiguration, SamplingConfig, ArchitectureConfig]):
        self.configuration = configuration

    def execute(self):
        self.suppress_rdkit_warnings()
        if self.configuration.run_type == "reinforcement":
            learning_scenario = self._create_reinforcement_learning()
            learning_scenario.execute()
        elif self.configuration.run_type == "sampling":
            learning_scenario = self._create_sampling_instance()
            learning_scenario.execute()
        else:
            learning_scenario = self._create_training_instance()

    def suppress_rdkit_warnings(self):
        RDLogger.DisableLog('rdApp.*')

    def _create_reinforcement_learning(self):
        model_type = ModelTypeEnum()
        model_mode = ModelModeEnum()
        agent_config = ModelConfiguration(model_type=model_type.MOL2MOL, model_mode=model_mode.INFERENCE,
                                          model_file_path=self.configuration.model_path)
        agent = GenerativeModel(agent_config)

        prior_config = ModelConfiguration(model_type=model_type.MOL2MOL, model_mode=model_mode.INFERENCE,
                                          model_file_path=self.configuration.model_path)
        prior = GenerativeModel(prior_config)
        learning_scenario = LearningScenario(agent, prior, self.configuration)
        return learning_scenario

    def _create_sampling_instance(self):
        model_type = ModelTypeEnum()
        model_mode = ModelModeEnum()
        agent_config = ModelConfiguration(model_type=model_type.MOL2MOL, model_mode=model_mode.INFERENCE,
                                          model_file_path=self.configuration.model_path)
        agent = GenerativeModel(agent_config)
        learning_scenario = Sampling(agent, self.configuration)
        return learning_scenario

    def _create_training_instance(self):
        trainer = TransformerTrainer(self.configuration)
        learning_scenario = trainer.execute()
        return learning_scenario
