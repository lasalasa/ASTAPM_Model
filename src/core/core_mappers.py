#------------- HFACS Mapping ----------
HFACS_DICTIONARY = {

    "hfacs": {
        # Unmapped
        -1: ('Unmapped', 'Unmapped', 'Unmapped', 'Unmapped'),
        # Unsafe Acts
        111: ('Level 1', 'Unsafe acts', 'Errors', 'Decision Errors'),
        112: ('Level 1', 'Unsafe acts', 'Errors', 'Skill-based Errors'),
        113: ('Level 1', 'Unsafe acts', 'Errors', 'Perceptual Errors'),
        121: ('Level 1', 'Unsafe acts', 'Violations', 'Routine Violations'),
        122: ('Level 1', 'Unsafe acts', 'Violations', 'Exceptional Violations'),

        # Preconditions for Unsafe Acts/Environmental Factors
        211: ('Level 2', 'Preconditions for Unsafe Acts', 'Environmental Factors', 'Physical Environment'),
        212: ('Level 2', 'Preconditions for Unsafe Acts', 'Environmental Factors', 'Technological Environment'),

        # Preconditions for Unsafe Acts/Conditions of Operators
        221: ('Level 2', 'Preconditions for Unsafe Acts', 'Conditions of Operators', 'Adverse Mental State'),
        222: ('Level 2', 'Preconditions for Unsafe Acts', 'Conditions of Operators', 'Adverse Physiological State'),
        223: ('Level 2', 'Preconditions for Unsafe Acts', 'Conditions of Operators', 'Physical Limitations'),

        # Preconditions for Unsafe Acts/'Personnel Factors
        231: ('Level 2', 'Preconditions for Unsafe Acts', 'Personnel Factors', 'Crew Resource Management'),
        232: ('Level 2', 'Preconditions for Unsafe Acts', 'Personnel Factors', 'Personal Readiness'),
        
        # Unsafe Supervision
        321: ('Level 3', 'Unsafe Supervision', 'Unsafe Supervision', 'Planned Inappropriate Operations'),
        311: ('Level 3', 'Unsafe Supervision', 'Unsafe Supervision', 'Inadequate Supervision'),
        331: ('Level 3', 'Unsafe Supervision', 'Unsafe Supervision', 'Failure to Correct Known Problems'),
        341: ('Level 3', 'Unsafe Supervision', 'Unsafe Supervision', 'Supervisory Violations'),
        
        # Organizational Influences
        411: ('Level 4', 'Organizational Influences', 'Organizational Influences', 'Organizational Climate'),
        421: ('Level 4', 'Organizational Influences', 'Organizational Influences', 'Organizational Process'),
        431: ('Level 4', 'Organizational Influences', 'Organizational Influences', 'Resource Management')
    },
    
    # AST-APM Taxonomy
    "hfacs_balance": {
        # Unmapped
        -1: ('Unmapped', 'Unmapped', 'Unmapped', 'Unmapped'),
        # Unsafe Acts
        111: ('Level 1', 'Unsafe acts', 'Errors/Violations', 'Decision/Skill-based/Perceptual Errors/Violations'),

        # Preconditions for Unsafe Acts
        # Preconditions/Physical Environment
        211: ('Level 2', 'Preconditions for Unsafe Acts', 'Environmental Factors', 'Physical Environment'),
        # Preconditions/Technological Environment
        212: ('Level 2', 'Preconditions for Unsafe Acts', 'Environmental Factors', 'Technological Environment'),
        # Preconditions/Conditions of Operators
        221: ('Level 2', 'Preconditions for Unsafe Acts', 'Conditions of Operators', 'Adverse Mental State/Adverse Physiological State/Physical Limitations'),
        # Preconditions/Personnel Factors
        231: ('Level 2', 'Preconditions for Unsafe Acts', 'Personnel Factors', 'Crew Resource Management/Personal Readiness'),
        
        # Unsafe Supervision
        321: ('Level 3', 'Unsafe Supervision', 'Unsafe Supervision', 'Planned Inappropriate Operations/Inadequate Supervision/Failure to Correct Known Problems/Supervisory Violations'),
        
        # Organizational Influences
        411: ('Level 4', 'Organizational Influences', 'Organizational Influences', 'Organizational Climate/Organizational Process/Resource Management'),
    }
}
#------------- HFACS Mapping ----------

AST_APM_DICTIONARY = {
    -1: ('Unmapped', 'Unmapped', 'Unmapped', 'Unmapped'),

    111: ('Level 1', 'Unsafe acts', 'Unsafe acts', 'Unsafe acts'),

    211: ('Level 2', 'Preconditions/Physical Environment', 'Preconditions/Physical Environment', 'Preconditions/Physical Environment'),
    212: ('Level 2', 'Preconditions/Technological Environment', 'Preconditions/Technological Environment', 'Preconditions/Technological Environment'),
    221: ('Level 2', 'Preconditions/Conditions of Operators', 'Preconditions/Conditions of Operators', 'Preconditions/Conditions of Operators'),
    231: ('Level 2', 'Preconditions/Personnel Factors', 'Preconditions/Personnel Factors', 'Preconditions/Personnel Factors', 'Preconditions/Personnel Factors'),

    321: ('Level 3', 'Unsafe Supervision', 'Unsafe Supervision', 'Unsafe Supervision'),

    411: ('Level 4', 'Organizational Influences', 'Organizational Influences', 'Organizational Influences')
}

#------------- ASRS Manual Mapping ----------
AUTO_LABELING_DICTIONARY = {
    "asrs": {
        "Aircraft": 212,                                        # Preconditions for Unsafe Acts => Technological Environment
        "Airport": 211,                                         # Preconditions for Unsafe Acts => Physical Environment
        "Airspace Structure": 212,                              # Preconditions for Unsafe Acts => Technological Environment
        "ATC Equipment / Nav Facility / Buildings": 212,        # Preconditions for Unsafe Acts => Technological Environment
        "Chart Or Publication": 421,                            # Organizational Influences => Organizational Process
        "Company Policy": 411,                                  # Organizational Influences => Organizational Climate
        "Environment - Non Weather Related": 211,               # Preconditions for Unsafe Acts => Physical Environment
        "Equipment / Tooling": 212,                             # Preconditions for Unsafe Acts => Technological Environment
        
        # Human Factors
        "Human Factors:Communication Breakdown": 231,     # Preconditions for Unsafe Acts => Personnel Factors => Crew Resource Management)
        "Human Factors:Confusion": 113,                   # Perceptual Errors
        "Human Factors:Distraction": 112,                 # Skill-based Errors
        "Human Factors:Fatigue": 221,                     # Adverse Mental State
        "Human Factors:Human-Machine Interface": 212,     # Technological Environment
        # "Other / Unknown": (-1, -1, -1),  # General Error (Unspecified)
        "Human Factors:Physiological - Other": 222,       # Adverse Physiological State
        "Human Factors:Situational Awareness": 221,       # Conditions of Operators → Adverse Mental State
        "Human Factors:Time Pressure": 221,               # Adverse Mental State
        "Human Factors:Training / Qualification": 112,    # Personal Readiness
        "Human Factors:Troubleshooting": 112,             # Skill-based Errors
        "Human Factors:Workload": 221,                 # Adverse Mental State
        
        "Human Factors": 111,                                    # Unsafe Acts => Errors or Violations
        "Incorrect / Not Installed / Unavailable Part": 431,    # Organizational Influences => Resource Management
        "Logbook Entry": 421,                                   # Organizational Influences => Organizational Process
        "Manuals": 421,                                         # Organizational Influences => Organizational Process
        "MEL": 421,                                             # Organizational Influences => Organizational Process
        "Procedure": 421,                                       # Organizational Influences => Organizational Process
        "Software and Automation": 212,                         # Preconditions for Unsafe Acts => Technological Environment
        "Staffing": 431,                                        # Organizational Influences => Resource Management
        "Weather": 211                                          # Preconditions for Unsafe Acts => Physical Environment
    },
    "asrs_balance": {

        "Airport": 211,
        "Environment - Non Weather Related": 211, 
        "Weather": 211,

        "Aircraft": 212,
        "Airspace Structure": 212,
        "Software and Automation": 212,
        "ATC Equipment / Nav Facility / Buildings": 212,
        "Equipment / Tooling": 212,
        "Human Factors:Human-Machine Interface": 212, 
        
        # Human Factors
        "Human Factors:Fatigue": 221, 
        "Human Factors:Physiological - Other": 221,
        "Human Factors:Situational Awareness": 221, 
        "Human Factors:Time Pressure": 221,  
        "Human Factors:Workload": 221,  

        "Human Factors:Communication Breakdown": 231,
        "Human Factors:Training / Qualification": 231, 

        "Human Factors:Troubleshooting": 111,
        "Human Factors:Confusion": 111,
        "Human Factors:Distraction": 111,
        "Human Factors": 111,

        "Chart Or Publication": 411,
        "Company Policy": 411,    
        "Incorrect / Not Installed / Unavailable Part": 411,
        "Logbook Entry": 411,
        "Manuals": 411,
        "MEL": 411,
        "Procedure": 411,
        "Staffing": 411        
    },
    "ntsb": {
        # Unsafe Acts => Skill-based Errors
        "Personnel issues-Task performance": 112,
        # Unsafe Acts => Decision Errors
        "Personnel issues-Action/decision": 111,
        # Unsafe Acts => Perceptual Errors
        "Personnel issues-Psychological-Perception/orientation/illusion": 113,
        # Unsafe Acts => Violations
        "Personnel issues-Miscellaneous-Intentional act": 121,

        # Preconditions for Unsafe Acts => Physical Environment
        'Environmental issues-Physical environment': 211,
        'Environmental issues-Conditions/weather/phenomena-Turbulence': 211,
        'Environmental issues-Task environment': 211,

        # Preconditions for Unsafe Acts => Technological Environment
        'Environmental issues-Operating environment': 212,
        'Aircraft-Aircraft systems': 212,

        # Preconditions for Unsafe Acts => Adverse Mental State
        'Personnel issues-Physical-Alertness/Fatigue': 221,
        'Personnel issues-Psychological-Attention/monitoring': 221,
        'Personnel issues-Psychological-Personality/attitude': 221,
        'Personnel issues-Psychological-Mental/emotional state': 221,

        # Preconditions for Unsafe Acts => Adverse Physiological State
        'Personnel issues-Psychological': 222,
        'Personnel issues-Physical-Impairment/incapacitation': 222,
        'Personnel issues-Physical-Health/Fitness': 222,

        # Preconditions for Unsafe Acts => Physical Limitations
        'Personnel issues-Physical-Sensory ability/limitation': 223,

        # Preconditions for Unsafe Acts => Crew Resource Management
        'Lack of communication': 231,

        # Preconditions for Unsafe Acts => Personal Readiness
        "Personnel issues-Experience/knowledge": 232,

        # Organizational Influences => Organizational Climate
        'Organizational issues-Management-Culture': 411,
        'Organizational issues-Management-Communication (organizational)': 411,

        # Organizational Influences =>  Organizational Process
        'Organizational issues-Management-Scheduling':  421,
        'Organizational issues-Management-Policy/procedure':  421,
        'Organizational issues-Management-(general)-(general)-Operator':  421,
        'Organizational issues-Support/oversight/monitoring': 421,

        # Organizational Influences =>  Resource Management
        'Organizational issues-Management-Resources': 431,
        'Organizational issues-Development-Selection/certification/testing': 431,
    },
    "ntsb_balance": {
        # Unsafe Acts => Skill-based Errors
        "Personnel issues-Task performance": 111,
        # Unsafe Acts => Decision Errors
        "Personnel issues-Action/decision": 111,
        # Unsafe Acts => Perceptual Errors
        "Personnel issues-Psychological-Perception/orientation/illusion": 111,
        # Unsafe Acts => Violations
        "Personnel issues-Miscellaneous-Intentional act": 111,

        # Preconditions for Unsafe Acts => Physical Environment
        'Environmental issues-Physical environment': 211,
        'Environmental issues-Conditions/weather/phenomena-Turbulence': 211,
        'Environmental issues-Task environment': 211,

        # Preconditions for Unsafe Acts => Technological Environment
        'Environmental issues-Operating environment': 212,
        'Aircraft-Aircraft systems': 212,

        # Preconditions for Unsafe Acts => Adverse Mental State
        'Personnel issues-Physical-Alertness/Fatigue': 221,
        'Personnel issues-Psychological-Attention/monitoring': 221,
        'Personnel issues-Psychological-Personality/attitude': 221,
        'Personnel issues-Psychological-Mental/emotional state': 221,
        # Preconditions for Unsafe Acts => Adverse Physiological State
        'Personnel issues-Psychological': 221,
        'Personnel issues-Physical-Impairment/incapacitation': 221,
        'Personnel issues-Physical-Health/Fitness': 221,
        # Preconditions for Unsafe Acts => Physical Limitations
        'Personnel issues-Physical-Sensory ability/limitation': 221,

        # Preconditions for Unsafe Acts => Crew Resource Management
        'Lack of communication': 231,
        # Preconditions for Unsafe Acts => Personal Readiness
        "Personnel issues-Experience/knowledge": 231,

        # Organizational Influences => Organizational Climate
        'Organizational issues-Management-Culture': 411,
        'Organizational issues-Management-Communication (organizational)': 411,
        # Organizational Influences =>  Organizational Process
        'Organizational issues-Management-Scheduling':  411,
        'Organizational issues-Management-Policy/procedure':  411,
        'Organizational issues-Management-(general)-(general)-Operator':  411,
        'Organizational issues-Support/oversight/monitoring': 411,
        # Organizational Influences =>  Resource Management
        'Organizational issues-Management-Resources': 411,
        'Organizational issues-Development-Selection/certification/testing': 411,
    }
}