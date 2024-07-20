import os
import torch
import argparse
from whisperspeech.pipeline import Pipeline

#---------------------------------------------
code_dir = os.getcwd()
parser = argparse.ArgumentParser(description='Code to convert ext to speech using WhisperSpeech')

parser.add_argument(
  '--txt', 
  type=str,
  help='Input text to be converted to speech.', 
  default='Testing text to speech using whisperspeech'
  )

parser.add_argument(
  '--outfile',
  type=str,
  help='Path to save the output.', 
  default='results/r.wav'
  )

parser.add_argument(
  '--clone', 
  type=str,
  help='Path of reference speaker audio to be cloned.', 
  )

parser.add_argument(
  '--speaker', type=int, 
					help='Select default speaker from input folder. 1,2,3 etc..', default=0)  

args = parser.parse_args()
#---------------------------------------------

def main():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  pipe = Pipeline(
      s2a_ref=f'{code_dir}/models/whisp/s2a-q4-tiny-en+pl.model', 
      t2s_ref=f'{code_dir}/models/whisp/t2s-small-en+pl.model', 
      voc_ref=f'{code_dir}/models/vocos',    
      spbr_ref=f'{code_dir}/models/speechbrain',
      device=device
  )
  if args.speaker or args.clone:
    if args.speaker:
      clone_file=f'{code_dir}/input/{args.speaker}.wav'
      pipe.generate_to_file(
        args.outfile, 
        args.txt, 
        lang='en', 
        cps=10.5, 
        speaker=args.clone
        ) 
    else:                  
      pipe.generate_to_file(
        args.outfile, 
        args.txt, 
        lang='en', 
        cps=10.5, 
        speaker=args.clone
        )  
  else:
      pipe.generate_to_file(args.outfile, args.txt)
      
#---------------------------------------------
if __name__ == '__main__':
	main()
