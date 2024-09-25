# HFACS Key Dictionary
HFACS_DICTIONARY = {

    # "hfacs": {
        # Unmapped
    -1: ('Unmapped', 'Unmapped', 'Unmapped', ''),
    # Unsafe Acts
    110: ('Level 1', 'Unsafe acts', 'Errors/Violations', 'Errors/Violations'), # 'Decision/Skill-based/Perceptual Errors/Violations'
    # Unsafe Acts / Errors
    111: ('Level 1', 'Unsafe acts', 'Errors', 'Skill-based Errors'),
    112: ('Level 1', 'Unsafe acts', 'Errors', 'Decision Errors'),
    113: ('Level 1', 'Unsafe acts', 'Errors', 'Perceptual Errors'),
    # Unsafe Acts / Violations
    120: ('Level 1', 'Unsafe acts', 'Violations', 'Routine Violations/Exceptional Violations'),
    121: ('Level 1', 'Unsafe acts', 'Violations', 'Routine Violations'),
    122: ('Level 1', 'Unsafe acts', 'Violations', 'Exceptional Violations'),

    # Preconditions for Unsafe Acts
    210: ('Level 2', 'Preconditions for Unsafe Acts', 'Environmental Factors', 'Physical / Technological Environment'),
    # Preconditions for Unsafe Acts/Environmental Factors
    211: ('Level 2', 'Preconditions for Unsafe Acts', 'Environmental Factors', 'Physical Environment'),
    212: ('Level 2', 'Preconditions for Unsafe Acts', 'Environmental Factors', 'Technological Environment'),
    # Preconditions for Unsafe Acts/Conditions of Operators
    220: ('Level 2', 'Preconditions for Unsafe Acts', 'Conditions of Operators', 'Conditions of Operators'), # 'Adverse Mental State/Adverse Physiological State/Physical Limitations'
    221: ('Level 2', 'Preconditions for Unsafe Acts', 'Conditions of Operators', 'Adverse Mental State'),
    222: ('Level 2', 'Preconditions for Unsafe Acts', 'Conditions of Operators', 'Adverse Physiological State'),
    223: ('Level 2', 'Preconditions for Unsafe Acts', 'Conditions of Operators', 'Physical/Mental Limitations'),
    # Preconditions for Unsafe Acts/'Personnel Factors
    230: ('Level 2', 'Preconditions for Unsafe Acts', 'Personnel Factors', 'Personnel Factors'), # 'Crew Resource Management/Personal Readiness'
    231: ('Level 2', 'Preconditions for Unsafe Acts', 'Personnel Factors', 'Crew Resource Management'),
    232: ('Level 2', 'Preconditions for Unsafe Acts', 'Personnel Factors', 'Personal Readiness'),
    
    # Unsafe Supervision
    310: ('Level 3', 'Unsafe Supervision', 'Unsafe Supervision', 'Unsafe Supervision'), # Planned Inappropriate Operations/Inadequate Supervision/Failure to Correct Known Problems/Supervisory Violations
    311: ('Level 3', 'Unsafe Supervision', 'Unsafe Supervision', 'Inadequate Supervision'),
    321: ('Level 3', 'Unsafe Supervision', 'Unsafe Supervision', 'Planned Inappropriate Operations'),
    331: ('Level 3', 'Unsafe Supervision', 'Unsafe Supervision', 'Failure to Correct Known Problems'),
    341: ('Level 3', 'Unsafe Supervision', 'Unsafe Supervision', 'Supervisory Violations'),
    
    # Organizational Influences
    410: ('Level 4', 'Organizational Influences', 'Organizational Influences', 'Organizational Influences'), # Organizational Climate/Organizational Process/Resource Management
    411: ('Level 4', 'Organizational Influences', 'Organizational Influences', 'Organizational Climate'),
    421: ('Level 4', 'Organizational Influences', 'Organizational Influences', 'Organizational Process'),
    431: ('Level 4', 'Organizational Influences', 'Organizational Influences', 'Resource Management')
}

# HFACS Mapping
hfacs_mapping = {
    'Unsafe Acts': {
        # Errors/Skill-based
        111: [
            # ASRS
            'Distraction', 
            'Troubleshooting',

            # NTSB
            'Personnel issues-Task performance'
        ],
        # Errors/Decision
        112: [
            # NTSB
            'Personnel issues-Action/decision'
        ],
        # Errors/Perceptual
        113: [
            # ASRS
            'confusion', 
            'Ambiguous',

            # NTSB
            'Personnel issues-Psychological-Perception/orientation/illusion'
        ],
        # Violations
        120: [

            # NTSB
            'Personnel issues-Miscellaneous-Intentional act'
        ]
    },
    'Preconditions for Unsafe Acts': {
        # Physical Environment
        211: [
            # ASRS
            'Weather', 
            'Environment - Non Weather Related',

            # NTSB
            'Environmental issues-Physical environment',
            'Environmental issues-Conditions/weather/phenomena-Turbulence',
            'Environmental issues-Task environment'
        ],
        # Technological Environment
        212: [
            # ASRS
            'Software and Automation', 
            'Incorrect / Not Installed / Unavailable Part', 
            
            # ASRS and NTSB
            'Aircraft',
            
            # NTSB
            'Environmental issues-Operating environment',
            # 'Aircraft-Aircraft systems'
        ],
        # 'Conditions of Operators/Adverse Mental States
        221: [
            # ASRS
            'Situational Awareness', 
            'time pressure', 
            'distraction', 
            'fatigue',
            'Workload',
            'Human-machine interface'

            # NTSB
            'Personnel issues-Physical-Alertness/Fatigue',
            'Personnel issues-Psychological-Attention/monitoring',
            'Personnel issues-Psychological-Personality/attitude',
            'Personnel issues-Psychological-Mental/emotional state',
        ],
        # Conditions of Operators/Adverse Physiological States
        222: [
            # ASRS
            'Physiological - Other'

            # NTSB
            'Personnel issues-Psychological',
            'Personnel issues-Physical-Impairment/incapacitation',
            'Personnel issues-Physical-Health/Fitness',
        ],
        # Conditions of Operators/Physical/Mental Limitations
        223: [
            # NTSB
            'Personnel issues-Physical-Sensory ability/limitation'
        ],
        # Personnel Factors/Crew Resource Management
        231: [
            # ASRS
            'communication breakdown',

            # NTSB
            'Lack of communication',
        ],
        # Personnel Factors/Personal Readiness
        232: [
            # ASRS
            'Training / Qualification',

            # NTSB
            'Personnel issues-Experience/knowledge'
        ]
    },
    'Unsafe Supervision': {
        # Unsafe Supervision/Inadequate Supervision
        311: [
            # ASRS
            'Logbook Entry'
        ],
        # Unsafe Supervision/Planned Inappropriate Operations
        321:[],
        # Unsafe Supervision/Failure to Correct Known Problems
        331: [],
        # Unsafe Supervision/Supervisory Violations
        341: []
    },
    'Organizational Influences': {
        # Organizational Influences/Organizational Climate
        411: [
            # ASRS
            'Procedure', 
            'Manuals', 
            'Airport', 
            'ATC Equipment / Nav Facility / Buildings', 
            'Chart Or Publication',

            # NTSB
            'Organizational issues-Management-Scheduling',
            'Organizational issues-Management-Policy/procedure',
            'Organizational issues-Management-(general)-(general)-Operator',
            'Organizational issues-Support/oversight/monitoring',
        ],
        # Organizational Influences/Operational Processes
        421: [
            # ASRS
            'Staffing', 
            'Equipment / Tooling',
            'MEL'

            # NTSB
            'Organizational issues-Management-Resources',
            'Organizational issues-Development-Selection/certification/testing'
        ],
         # Organizational Influences/Resource Management
        431: [
            # ASRS
            'Company Policy', 
            'Airspace Structure',

            # NTSB
            'Organizational issues-Management-Culture',
            'Organizational issues-Management-Communication (organizational)',
        ]
    }
}

# HFACS Mapping Balancing data
hfacs_mapping_balance = {
    # Level 01
    'Unsafe Acts': {
        110: [
            # ASRS
            'Distraction', 
            'Troubleshooting', 
            'confusion', 
            'Ambiguous',

            # NTSB
            'Personnel issues-Task performance', 
            'Personnel issues-Action/decision', 
            'Personnel issues-Psychological-Perception/orientation/illusion',
            'Personnel issues-Miscellaneous-Intentional act'
        ],
    },
    # Level 02
    'Preconditions for Unsafe Acts': {
        # Physical Environment
        211: [
            # ASRS
            'Weather', 
            'Environment - Non Weather Related',

            # NTSB
            'Environmental issues-Physical environment',
            'Environmental issues-Conditions/weather/phenomena-Turbulence',
            'Environmental issues-Task environment'
        ],
        # Technological Environment
        212: [
            # ASRS
            'Software and Automation', 
            'Incorrect / Not Installed / Unavailable Part', 
            
            # ASRS and NTSB
            'Aircraft',
            
            # NTSB
            'Environmental issues-Operating environment',
            # 'Aircraft-Aircraft systems'
        ],
        # Conditions of Operators
        220: [
            # ASRS
            'Situational Awareness', 
            'time pressure', 
            'distraction', 
            'fatigue',
            'Workload',
            'Human-machine interface',
            'Physiological - Other',

            # NTSB
            'Personnel issues-Physical-Alertness/Fatigue',
            'Personnel issues-Psychological-Attention/monitoring',
            'Personnel issues-Psychological-Personality/attitude',
            'Personnel issues-Psychological-Mental/emotional state',
            
            'Personnel issues-Psychological',
            'Personnel issues-Physical-Impairment/incapacitation',
            'Personnel issues-Physical-Health/Fitness',
            
            'Personnel issues-Physical-Sensory ability/limitation'
        ],
        # Personnel Factors
        230: [
            # ASRS
            'communication breakdown',
            'Training / Qualification',

            # NTSB
            'Lack of communication',
            'Personnel issues-Experience/knowledge'
        ]
    },
    # Level 04
    # Organizational Influences
    'Organizational Influences': {
        410: [
            # ASRS
            'Company Policy', 
            'Airspace Structure',
            'Procedure', 'Manuals', 'Airport', 'ATC Equipment / Nav Facility / Buildings', 'Chart Or Publication',
            'Staffing', 'Equipment / Tooling', 'MEL',

            # NTSB
            'Organizational issues-Management-Culture',
            'Organizational issues-Management-Communication (organizational)',
            'Organizational issues-Management-Scheduling',
            'Organizational issues-Management-Policy/procedure',
            'Organizational issues-Management-(general)-(general)-Operator',
            'Organizational issues-Support/oversight/monitoring',
            'Organizational issues-Management-Resources',
            'Organizational issues-Development-Selection/certification/testing'
        ]
    }
}

# HFACS Map Dictionary
HFACS_MAPPING_DICTIONARY = {
    # "hfacs": hfacs_categories_mapping,
    "hfacs_mapping": hfacs_mapping,
    "hfacs_mapping_balance": hfacs_mapping_balance
}