from __future__ import annotations

from typing import Self

from ..epiconfig import EpiConfig
from ..columnregistration import ColumnRegistry

from .orchestrator_helpers import (
    EpiDataReader, EpiDataHarmonizer, EpiDataProcessor, 
    EpiFeatureBuilder, EpiDataTransformer, EpiDataFinalizer
)
from .containers import (
    RawEpiData, HarmonizedEpiData, ContextEpiData, ProcessedEpiData, 
    FeatureEpiData, TransformedEpiData, FinalizedEpiData
)
from .validators import (
    RawValidator, HarmonizedValidator, ContextValidator, ProcessedValidator, 
    FeatureValidator, TransformedValidator, FinalizedValidator
)

from .utils import MissingEpiDataContainer

class EpiDataOrchestrator:
    """
    Main Orchestrator class that transforms raw data, into a shared data set that all
    models will be using. In the pipeline that ``EpiDataOrchestrator`` orchestrates, 
    data is loaded, harmonized, processed and transformed (and normalized). These steps 
    are executed by the helper classes of ``EpiDataOrchestrator``. Before storing the
    data classes into attributes (see section 'Attributes'), they are validated using
    the specific validator.

    Parameters
    ----------
    config : EpiConfig
        Large configuration class that dictates which data to load.            

    Methods
    -------
    There is a single method for each stage of the pipeline. There is also a single 
    method that calls on all of these. Note that the ``load()`` method takes 
    particularly long. As such, there is an alternative ``from_raw()`` method.
    Each of these methods additionally validates the outputted data class, by throwing
    it into the respective validator class.
    Each of these method returns ``EpiDataOrchestrator``, to allow chaining.

    ``build()``
        The one method that calls all others. 
    ``load()``
        Load the raw data: let ``EpiDataReader`` create ``RawEpiData``.
    ``harmonize()``
        Harmonize the raw data: let ``EpiDataHarmonizer`` create ``HarmonizedEpiData`` 
        and ``ContextEpiData``.
    ``process()``
        Process the harmonized data: let ``EpiDataProcessor`` create 
        ``ProcessedEpiData``.
    ``build_features()``
        Build features from the processed data: let ``EpiFeatureBuilder`` create 
        ``FeatureEpiData``.    
    ``normalize()``
        Transform the feature data: let ``EpiDataTransformer`` create 
        ``TransformedEpiData``.    
    ``finalize()``
        Finish the data orchestration: let ``EpiDataFinalizer`` create 
        ``FinalizedEpiData``.

    Attributes
    ----------
    The intermediate data containers are stored into attributes. Note that when one
    is called that does not contain a container, an exception will be raised.
    ``data_raw``
    ``data_context``
    ``data_harmonized``
    ``data_processed``
    ``data_features``
    ``data_normalized``
    ``data_final``

    Downstream
    ----------
    The final product of ``EpiDataOrchestrator`` is the ``data_final`` attribute. This 
    ``FinalizedEpiData`` contains attributes ``data_denorm`` and ``data``. These are 
    the starting point for ``BaseLineDataBuilder`` and ``GraphDataBuilder``: the model-
    specific data loaders.
    """
    def __init__(self, config: EpiConfig):
        self.config         = config

        # Initialize column registration
        self.column_registration = ColumnRegistry()
        self.column_registration.add_column(
            config.temporal_column, 
            'context'
        )
        self.column_registration.add_column(
            self.config.id_column, 
            'context'
        )   
        self.column_registration.add_column(
            'target', 
            'target',
            transformation =  True,
            transformation_group = 'self'
        )   

        # Store results at each stage
        self._data_raw       = None
        self._data_harmonized= None
        self._data_context   = None
        self._data_processed = None
        self._data_feature   = None
        self._data_normalized= None
        self._data_final     = None
          
    def load(self) -> EpiDataOrchestrator:
        """Load raw data from files"""
        self.reader         = EpiDataReader(self.config)
        rawepidata          = self.reader.orchestrate()
        self._data_raw      = rawepidata

        RawValidator(self.config, rawepidata).validate()

        return self
    
    @classmethod
    def from_raw(cls, config: EpiConfig, 
                data_raw: RawEpiData) -> Self:
        """
        Bypass load_raw.
        Use when RawEpiData is already available (e.g. cross-validation).
        """
        orch = cls(config)
        orch._data_raw = data_raw
        return orch
    
    def harmonize(self) -> Self:
        """Harmonize data on NUTS-level"""     
        self.harmonizer                             = EpiDataHarmonizer(self.config)   
        harmepidata, contextepidata                 = self.harmonizer.orchestrate(self.data_raw)
        self._data_harmonized, self._data_context   = harmepidata, contextepidata        

        HarmonizedValidator(self.config, harmepidata).validate()
        ContextValidator(self.config, contextepidata).validate()

        return self    

    def process(self) -> Self:
        """Preprocess the harmonized data"""
        self.processor = EpiDataProcessor(
            self.config, 
            self.data_context.temporal_summary
        )
        processed_data       = self.processor.orchestrate(self.data_harmonized)
        self._data_processed = processed_data

        ProcessedValidator(self.config, processed_data).validate()            
        return self

    def build_features(self) -> Self:
        """build features. Note that this method adjusts self.column_registry."""
        self.feature_builder= EpiFeatureBuilder(self.config, self.column_registration, self.data_context.temporal_summary)
        feature_data        = self.feature_builder.orchestrate(self.data_processed)
        self._data_feature  = feature_data

        FeatureValidator(self.config, self.column_registration, feature_data).validate()          
        return self        
   
    def normalize(self) -> Self:
        """normalize data"""
        self.normalizer = EpiDataTransformer(
            self.config, 
            self.column_registration,
            self.data_context.temporal_summary
        )
        normalized_data = self.normalizer.orchestrate(self.data_feature)
        self._data_normalized = normalized_data
        TransformedValidator(self.config, self.column_registration, normalized_data).validate()
        return self      

    def finalize(self) -> Self:
        """Finalize data."""
        self.finalizer      = EpiDataFinalizer(self.config, self.column_registration)
        final_data          = self.finalizer.orchestrate(self.data_normalized)
        self._data_final    = final_data
        FinalizedValidator(self.config, self.column_registration, self.data_raw, self.data_harmonized, final_data).validate()
        return self        

    def build(self) -> Self:
        """Execute full pipeline and return final dataset."""
        return (self
            .load()
            .harmonize()
            .process()
            .build_features()
            .normalize()
            .finalize()
        )
    
    def __repr__(self):
        stages = []
        if self._data_raw is not None:
            stages.append("raw")
        if self._data_harmonized is not None:
            stages.append("harmonized")            
        if self._data_processed is not None:
            stages.append('processed')
        if self._data_feature is not None:
            stages.append('features') 
        if self._data_normalized is not None:
            stages.append('normalized')
        if self._data_final is not None:
            stages.append('finalized')            
        
        return f"<EpiDataOrchestrator(disease={self.config.disease}, data stages={stages})>"
    
    @property
    def data_raw(self) -> RawEpiData:
        if not self._data_raw:
            raise MissingEpiDataContainer(datastage = 'data_raw', previous_method = 'load_raw')
        return self._data_raw    
    
    @property
    def data_context(self) -> ContextEpiData:
        if not self._data_context:
            raise MissingEpiDataContainer(datastage = 'data_context', previous_method = 'harmonize_raw')

        return self._data_context      

    @property
    def data_harmonized(self) -> HarmonizedEpiData:
        if not self._data_harmonized:
            raise MissingEpiDataContainer(datastage = 'data_harmonized', previous_method = 'harmonize_raw')

        return self._data_harmonized        
           
    @property
    def data_processed(self) -> ProcessedEpiData:
        if not self._data_processed:
            raise MissingEpiDataContainer(datastage = 'data_processed', previous_method = 'harmonize_raw')

        return self._data_processed    

    @property 
    def data_feature(self) -> FeatureEpiData:
        if not self._data_feature:
            raise MissingEpiDataContainer(datastage = 'data_feature', previous_method = 'data_processed')        

        return self._data_feature    
    
    @property
    def data_normalized(self) -> TransformedEpiData:
        if not self._data_normalized:
            raise MissingEpiDataContainer(datastage = 'data_normalized', previous_method = 'build_features')        

        return self._data_normalized    

    @property
    def data_final(self) -> FinalizedEpiData:
        if not self._data_final:
            raise MissingEpiDataContainer(datastage = 'data_final', previous_method = 'normalize')        

        return self._data_final         