import argparse
from whisperspeech.pipeline import Pipeline

#---------------------------------------------
parser = argparse.ArgumentParser(description='Code to convert ext to speech using WhisperSpeech')

parser.add_argument(
  '--txt', 
  type=str,
  help='Input text to be converted to speech.', 
  default='Hello! how are you doing today
)

parser.add_argument(
  '--speaker', 
  type=str,
  help='Path of reference speaker audio to be cloned.', 
  default='input/a.wav'
)

parser.add_argument(
  '--outfile',
  type=str,
  help='Path to save the output.', 
  default='results/r.wav'
)

args = parser.parse_args()
#---------------------------------------------
def main():
  pipe = Pipeline(
      s2a_ref=f'{p_ws_model}models/s2a-q4-tiny-en+pl.model', 
      t2s_ref=f'{p_ws_model}/t2s-small-en+pl.model', 
      voc_ref=p_ve_model,    
      spbr_ref=p_sb_model,
      device=device
  )
  
if __name__ == '__main__':
	main()
