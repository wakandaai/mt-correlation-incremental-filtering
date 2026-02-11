"""
Language Configuration for FLEURS-NLLB Correlation Analysis
Maps language names to FLEURS codes and NLLB codes
"""

# Language mapping for the 25 African + Global languages
# Format: {
#   'common_name': {
#       'fleurs_code': 'code_used_in_fleurs_dataset',
#       'nllb_code': 'code_used_in_nllb_model',
#       'iso_code': 'standard_iso_code'
#   }
# }

LANGUAGE_CONFIG = {
    'French': {
        'fleurs_code': 'fr_fr',
        'nllb_code': 'fra_Latn',
        'iso_code': 'fra',
        'priority': 'test'  # Use for initial testing
    },
    'Portuguese': {
        'fleurs_code': 'pt_br',
        'nllb_code': 'por_Latn',
        'iso_code': 'por',
        'priority': 'standard'
    },
    'Arabic': {
        'fleurs_code': 'ar_eg',
        'nllb_code': 'arb_Arab',
        'iso_code': 'ara',
        'priority': 'standard'
    },
    'Afrikaans': {
        'fleurs_code': 'af_za',
        'nllb_code': 'afr_Latn',
        'iso_code': 'afr',
        'priority': 'standard'
    },
    'Swahili': {
        'fleurs_code': 'sw_ke',
        'nllb_code': 'swh_Latn',
        'iso_code': 'swa',
        'priority': 'test'  # Use for initial testing
    },
    'Somali': {
        'fleurs_code': 'so_so',
        'nllb_code': 'som_Latn',
        'iso_code': 'som',
        'priority': 'standard'
    },
    'Hausa': {
        'fleurs_code': 'ha_ng',
        'nllb_code': 'hau_Latn',
        'iso_code': 'hau',
        'priority': 'standard'
    },
    'Amharic': {
        'fleurs_code': 'am_et',
        'nllb_code': 'amh_Ethi',
        'iso_code': 'amh',
        'priority': 'standard'
    },
    # Note: Plateau Malagasy might not be in FLEURS, using standard Malagasy
    'Malagasy': {
        'fleurs_code': 'mg_mg',
        'nllb_code': 'plt_Latn',  # Plateau Malagasy
        'iso_code': 'mlg',
        'priority': 'standard'
    },
    'Kinyarwanda': {
        'fleurs_code': 'rw_rw',  # May need to verify this
        'nllb_code': 'kin_Latn',
        'iso_code': 'kin',
        'priority': 'test'  # Use for initial testing
    },
    'Xhosa': {
        'fleurs_code': 'xh_za',
        'nllb_code': 'xho_Latn',
        'iso_code': 'xho',
        'priority': 'standard'
    },
    'Zulu': {
        'fleurs_code': 'zu_za',
        'nllb_code': 'zul_Latn',
        'iso_code': 'zul',
        'priority': 'standard'
    },
    'Chichewa': {
        'fleurs_code': 'ny_mw',  # Nyanja/Chichewa
        'nllb_code': 'nya_Latn',
        'iso_code': 'nya',
        'priority': 'standard'
    },
    'Sesotho': {
        'fleurs_code': 'st_za',
        'nllb_code': 'sot_Latn',
        'iso_code': 'sot',
        'priority': 'standard'
    },
    'Shona': {
        'fleurs_code': 'sn_zw',
        'nllb_code': 'sna_Latn',
        'iso_code': 'sna',
        'priority': 'standard'
    },
    'Igbo': {
        'fleurs_code': 'ig_ng',
        'nllb_code': 'ibo_Latn',
        'iso_code': 'ibo',
        'priority': 'standard'
    },
    'Yoruba': {
        'fleurs_code': 'yo_ng',
        'nllb_code': 'yor_Latn',
        'iso_code': 'yor',
        'priority': 'standard'
    },
    'Tigrinya': {
        'fleurs_code': 'ti_et',  # May need verification
        'nllb_code': 'tir_Ethi',
        'iso_code': 'tir',
        'priority': 'standard'
    },
    'Luganda': {
        'fleurs_code': 'lg_ug',
        'nllb_code': 'lug_Latn',
        'iso_code': 'lug',
        'priority': 'standard'
    },
    'Lingala': {
        'fleurs_code': 'ln_cd',
        'nllb_code': 'lin_Latn',
        'iso_code': 'lin',
        'priority': 'standard'
    },
    'Setswana': {
        'fleurs_code': 'tn_za',
        'nllb_code': 'tsn_Latn',
        'iso_code': 'tsn',
        'priority': 'standard'
    },
    'Wolof': {
        'fleurs_code': 'wo_sn',
        'nllb_code': 'wol_Latn',
        'iso_code': 'wol',
        'priority': 'standard'
    },
    'Bemba': {
        'fleurs_code': 'bem_zm',  # May not be in FLEURS
        'nllb_code': 'bem_Latn',
        'iso_code': 'bem',
        'priority': 'verify'  # Need to verify FLEURS support
    },
    'Fongbe': {
        'fleurs_code': 'fon_bj',  # May not be in FLEURS
        'nllb_code': 'fon_Latn',
        'iso_code': 'fon',
        'priority': 'verify'  # Need to verify FLEURS support
    },
    'English': {
        'fleurs_code': 'en_us',
        'nllb_code': 'eng_Latn',
        'iso_code': 'eng',
        'priority': 'reference'  # This is our target language
    }
}

# Get test languages (sw, rw for initial testing)
def get_test_languages():
    """Return languages marked for testing"""
    return {k: v for k, v in LANGUAGE_CONFIG.items() if v['priority'] == 'test'}

# Get all languages except those needing verification
def get_standard_languages():
    """Return all standard languages (excluding verify and reference)"""
    return {k: v for k, v in LANGUAGE_CONFIG.items() 
            if v['priority'] in ['test', 'standard']}

# Get all languages
def get_all_languages():
    """Return all languages except English (reference)"""
    return {k: v for k, v in LANGUAGE_CONFIG.items() if v['priority'] != 'reference'}

if __name__ == "__main__":
    print("Test Languages (for initial run):")
    for lang, config in get_test_languages().items():
        print(f"  {lang}: FLEURS={config['fleurs_code']}, NLLB={config['nllb_code']}")
    
    print(f"\nTotal Standard Languages: {len(get_standard_languages())}")
    print(f"Total All Languages: {len(get_all_languages())}")