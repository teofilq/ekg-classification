from pathlib import Path

# constante pentru foldere
CURRENT_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = CURRENT_DIR / 'data' / 'processed'
FILTERED_DATA_DIR = CURRENT_DIR / 'data' / 'filtered_sese'
RAW_DATA_DIR = CURRENT_DIR / 'data' /'raw'/ 'WFDBRecords'
PLOT_DIR = CURRENT_DIR / 'plots'
CLASSIFIER_DATA_DIR = CURRENT_DIR / 'data' / 'classifier'
MODEL_1_DIR = CURRENT_DIR / 'models'

DEBUG = False

# parametri semnale
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SAMPLE_RATE = 500  
NUM_SAMPLES = 5000  
NUM_LEADS = 12

# codurile bolilor cu denumirile lor
SNOMED_DICT = {'270492004': ('1AVB', '1 degree atrioventricular block'), 
               '195042002': ('2AVB', '2 degree atrioventricular block'),
               '54016002': ('2AVB1', '2 degree atrioventricular block(Type one)'),
               '28189009': ('2AVB2', '2 degree atrioventricular block(Type two)'),
               '27885002': ('3AVB', '3 degree atrioventricular block'),
               '251173003': ('ABI', 'atrial bigeminy'),
               '39732003': ('ALS', 'Axis left shift'),
               '284470004': ('APB', 'atrial\xa0premature\xa0beats'),
               '164917005': ('AQW', 'abnormal Q wave'),
               '47665007': ('ARS', 'Axis right shift'),
               '233917008': ('AVB', 'atrioventricular block'),
               '251199005': ('CCR', 'countercolockwise rotation'),
               '251198002': ('CR', 'colockwise rotation'),
               '428417006': ('ERV', 'Early repolarization of the ventricles'),
               '164942001': ('FQRS', 'fQRS Wave'),
               '698252002': ('IVB', 'Intraventricular block'),
               '426995002': ('JEB', 'junctional escape beat'),
               '251164006': ('JPT', 'junctional premature beat'),
               '164909002': ('LFBBB', 'left front bundle branch block'),
               '164873001': ('LVH', 'left ventricle hypertrophy'),
               '251146004': ('LVQRSAL', 'lower voltage QRS in all lead'),
               '251148003': ('LVQRSCL', 'lower voltage QRS in chest lead'),
               '251147008': ('LVQRSLL', 'lower voltage QRS in limb lead'),
               '164865005': ('MISW', 'Myocardial infraction in the side wall'),
               '164947007': ('PRIE', 'PR interval extension'),
               '164912004': ('PWC', 'P wave Change'),
               '111975006': ('QTIE', 'QT interval extension'),
               '446358003': ('RAH', 'right atrial hypertrophy'),
               '59118001': ('RBBB', 'right bundle branch block'),
               '89792004': ('RVH', 'right ventricle hypertrophy'),
               '429622005': ('STDD', 'ST drop down'),
               '164930006': ('STE', 'ST extension'),
               '428750005': ('STTC', 'ST-T Change'),
               '164931005': ('STTU', 'ST tilt up'),
               '164934002': ('TWC', 'T wave Change'),
               '59931005': ('TWO', 'T wave opposite'),
               '164937009': ('UW', 'U wave'),
               '11157007': ('VB', 'ventricular bigeminy'),
               '75532003': ('VEB', 'ventricular escape beat'),
               '13640000': ('VFW', 'ventricular fusion wave'),
               '17338001': ('VPB', 'ventricular premature beat'),
               '195060002': ('VPE', 'ventricular preexcitation'),
               '251180001': ('VET', 'ventricular escape trigeminy'),
               '195101003': ('SAAWR', 'Sinus Atrium to Atrial Wandering Rhythm'),
               '74390002': ('WPW', 'WPW'),
               '426177001': ('SB', 'Sinus Bradycardia'),
               '426783006': ('SR', 'Sinus Rhythm'),
               '164889003': ('AFIB', 'Atrial Fibrillation'),
               '427084000': ('ST', 'Sinus Tachycardia'),
               '164890007': ('AF', 'Atrial Flutter'),
               '427393009': ('SA', 'Sinus Irregularity'),
               '426761007': ('SVT', 'Supraventricular Tachycardia'),
               '713422000': ('AT', 'Atrial Tachycardia'),
               '233896004': ('AVNRT', 'Atrioventricular  Node Reentrant Tachycardia'),
               '233897008': ('AVRT', 'Atrioventricular Reentrant Tachycardia')}
