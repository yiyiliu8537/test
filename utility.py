import os
import soundfile as sf

def writeAudio( dest , audio_info , rate=16000):
    if not os.path.exists( dest ):
        os.makedirs( dest )
	sf.write(dest , audio_info , rate)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
