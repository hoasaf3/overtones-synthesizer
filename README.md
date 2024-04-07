# overtones-synthesizer

This a a fun project, aimed to synthesize different music instruments based of their overtones amplitude ratio.

To create a new instrument:
1. Find a recording of a single note (.WAV)
2. Call `fft_wavfile(wav_filename)`
3. Create a ratios list, similar to GUIAR_RATIOS, FLUTE_RATIOS or CLARINET_RATIOS by examining the results in `freqs_amps`
4. Replace `CLARINET_RATIOS` in line 107 with your new instrument's ratios.

To play the synthesizer, open VirtualPiano/index.html
