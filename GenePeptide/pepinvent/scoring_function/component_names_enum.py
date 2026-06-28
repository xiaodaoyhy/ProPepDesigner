from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentNames:
    MaxRingSize: str = "maximum_ring_size"
    MolecularWeight: str = "molecular_weight"
    MatchingSubstructure: str = "substructure_match"
    PredictiveModel_GIPR: str = 'predictive_model_GIPR'
    PredictiveModel_GLP1R: str = 'predictive_model_GLP1R'
    PredictiveModel_GCGR: str = 'predictive_model_GCGR'
    Model_AD: str = 'Model_AD'
    Lipophilicity: str = 'lipophilicity'
    CustomAlerts: str = 'custom_alerts'
    TotalScore: str = "total_score"  # there is no actual component corresponding to this type

ComponentNamesEnum = ComponentNames()