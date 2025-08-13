"""
Class name utilities - Do one thing: manage class mappings
Unix philosophy: Simple, reusable class name operations
"""

from typing import Dict, List, Set


def create_class_mapping() -> Dict[str, str]:
    """Create standard class name mappings. Single purpose function."""
    return {
        'normal': 'Normal_Tympanic_Membrane',
        'aom': 'Acute_Otitis_Media', 
        'chronic': 'Chronic_Suppurative_Otitis_Media',
        'chornic': 'Chronic_Suppurative_Otitis_Media',  # Handle typo
        'earwax': 'Earwax_Cerumen_Impaction',
        'buson': 'Earwax_Cerumen_Impaction',  # Alternative name
        'earventulation': 'Ear_Ventilation_Tube',
        'chrneftup': 'Ear_Ventilation_Tube',  # Alternative name
        'foreign': 'Foreign_Bodies',
        'yabancisim': 'Foreign_Bodies',  # Alternative name
        'otitis_externa': 'Otitis_Externa',
        'otitexterna': 'Otitis_Externa',  # Handle no underscore
        'tympanoskleros': 'Tympanoskleros_Myringosclerosis',
        'pseudo_membranes': 'Pseudo_Membranes',
        'pseduo_membranes': 'Pseudo_Membranes',  # Handle typo
        'pseduomembran': 'Pseudo_Membranes'  # Handle compact form
    }


def normalize_class_name(folder_name: str) -> str:
    """Normalize folder name to standard class name."""
    mapping = create_class_mapping()
    return mapping.get(folder_name.lower(), folder_name)


def create_unified_classes() -> Dict[str, int]:
    """Create unified class taxonomy mapping."""
    return {
        'Normal_Tympanic_Membrane': 0,
        'Earwax_Cerumen_Impaction': 1,
        'Cerumen_Impaction': 1,  # Alternative
        'Acute_Otitis_Media': 2,
        'Chronic_Suppurative_Otitis_Media': 3,
        'Otitis_Externa': 4,
        'Tympanoskleros_Myringosclerosis': 5,
        'Myringosclerosis': 5,  # Alternative
        'Ear_Ventilation_Tube': 6,
        'Tympanostomy_Tubes': 6,  # Alternative
        'Pseudo_Membranes': 7,
        'Foreign_Bodies': 8,
        'Foreign_Objects': 8  # Alternative
    }


def get_class_names() -> Dict[int, str]:
    """Get class index to name mapping."""
    return {
        0: 'Normal_Tympanic_Membrane',
        1: 'Earwax_Cerumen_Impaction', 
        2: 'Acute_Otitis_Media',
        3: 'Chronic_Suppurative_Otitis_Media',
        4: 'Otitis_Externa',
        5: 'Tympanoskleros_Myringosclerosis',
        6: 'Ear_Ventilation_Tube',
        7: 'Pseudo_Membranes',
        8: 'Foreign_Bodies'
    }


def get_valid_classes() -> Set[str]:
    """Get set of valid class names."""
    return set(create_unified_classes().keys())