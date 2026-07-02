"""
Alice Configuration Management
==============================

Central configuration management for all Alice systems.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
import json
from pathlib import Path


@dataclass
class AliceConfig:
    """Central configuration for Alice systems"""
    
    # Model configuration
    model_name: str = "alice1.1.0-v3"
    model_path: str = "outputs/alice1.1.0-v3/"
    config_path: str = "configs/alice1.1.0-v3.yaml"
    
    # Personality settings
    default_chaos_level: float = 0.5
    # Persona text removed — identity material, not included in the
    # public release (see LICENSE).
    personality_template: str = """You are Alice. [Persona directives not included in public release.]"""
    
    # Safety configuration (Winnie the Pooh)
    safety_enabled: bool = True
    heffalump_threshold: float = 0.6
    woozle_threshold: float = 0.9
    
    # Memory configuration (Alice 1.1.1)
    memory_enabled: bool = True
    memory_db_path: str = "alice/data/databases/alice_memory.db"
    session_memory_limit: int = 100
    long_term_memory_importance_threshold: float = 0.5
    
    # Avatar configuration (future)
    avatar_enabled: bool = False
    avatar_system: str = "live2d"  # or "vroid"
    tts_system: str = "coqui"      # or "elevenlabs"
    
    # Paths
    data_dir: str = "data/"
    outputs_dir: str = "outputs/"
    configs_dir: str = "configs/"
    documentation_dir: str = "documentation/"
    
    # Environment
    environment: str = "development"  # or "production"
    debug_mode: bool = True
    logging_level: str = "INFO"
    
    @classmethod
    def load_from_file(cls, config_file: str) -> 'AliceConfig':
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            return cls()  # Return default config
            
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(**data)
    
    def save_to_file(self, config_file: str):
        """Save configuration to YAML or JSON file"""
        config_path = Path(config_file)
        
        # Convert to dictionary
        data = self.__dict__.copy()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)
    
    def get_chaos_parameters(self, chaos_level: Optional[float] = None) -> Dict[str, Any]:
        """Get sampling parameters based on chaos level"""
        if chaos_level is None:
            chaos_level = self.default_chaos_level
            
        if chaos_level < 0.3:
            return {
                "temperature": 0.7,
                "top_p": 0.85,
                "top_k": 50,
                "max_tokens": 128,
                "mode": "helpful_bratty"
            }
        elif chaos_level < 0.7:
            return {
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 96,
                "mode": "sassy_menace"
            }
        else:
            return {
                "temperature": 1.3,
                "top_p": 0.95,
                "top_k": 30,
                "max_tokens": 64,
                "mode": "unhinged_gremlin"
            }
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        # Check required paths exist
        required_paths = [self.data_dir, self.outputs_dir, self.configs_dir]
        for path in required_paths:
            if not Path(path).exists():
                print(f"Warning: Required path does not exist: {path}")
                return False
        
        # Check chaos level bounds
        if not 0.0 <= self.default_chaos_level <= 1.0:
            print(f"Error: default_chaos_level must be between 0.0 and 1.0, got {self.default_chaos_level}")
            return False
            
        # Check safety thresholds
        if not 0.0 <= self.heffalump_threshold <= 1.0:
            print(f"Error: heffalump_threshold must be between 0.0 and 1.0, got {self.heffalump_threshold}")
            return False
            
        return True


# Default configuration instance
default_config = AliceConfig()