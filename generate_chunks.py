from pathlib import Path
import soundfile as sf
import pandas as pd
import numpy as np
import random
import os

def get_augmentation_files(folder:str, samplerate:int, accepted_file_extensions:list=[".wav"]) -> dict:
    '''This method scans a folder for audio files ending with accepted_file_extensions in lower or upper case.
    It then uses soundfile to read those files. The method returns a dictionary with the filenames as keys and
    numpy.ndarrays of the corresponding audio files as values. If an audio file doesnt meet the required sample
    rate and error is thrown.'''
    audio_list = dict()
    
    for file in sorted(folder.iterdir()):
        if file.is_file() and file.suffix.lower() in accepted_file_extensions:
            audio, sr = sf.read(file)
    
            assert sr == samplerate, (
                f"{file.name} uses {sr} Hz instead of {samplerate}."
            )
    
            audio_list[file.name] = audio

    return audio_list

def augment_audio(chunk_audio:np.ndarray, aug_audio:np.ndarray, aug_volume:float=1):
    '''This method takes an ndarray chunk_audio and an ndarray aug_audio. It then adds a random segment of aug_audio
    with the length of chunk_audio to chunk_audio. aug_volume is used as a factor on how much of aug_audio is added to
    chunk_audio. The method returns the augmented chunk_audio.'''

    if len(aug_audio) < len(chunk_audio): raise ValueError("Cant use audio file for augmentation shorter than window size.")
    
    start = np.random.randint(0, len(aug_audio) - len(chunk_audio) + 1)
    end = start + len(chunk_audio)

    aug_chunk_audio = chunk_audio + aug_volume * aug_audio[start:end]

    return aug_chunk_audio

def generate_chunks(
    t_labels:pd.DataFrame,
    output_folder:str,
    samplerate:int,
    data_augmentation:bool = False,
    window_size:float      = 2,
    overlap:float          = 0,
    init_offset:float      = 0
    ) -> pd.DataFrame:

    input_folder = Path("recordings")
    output_folder = Path(output_folder)

    # parameters for data augmentation.
    augmentation_folder = Path("augmentation")
    accepted_file_extensions = [".wav"]
    aug_volume = 1

    # if the output folder already exists load the chunk labels and append the new data. Otherwise create a new folder and dataframe.
    if os.path.isdir(output_folder):
        print("Output folder already exists. Appending chunks ...")
        c_labels_path = output_folder / "chunk_labels.csv"
        c_labels = pd.read_csv(c_labels_path, sep=";", index_col=0)   
        chunk_index = len(c_labels)
    else:
        print("Create output folder ...")
        output_folder.mkdir()
        c_labels = pd.DataFrame(columns=["chunk_idx", "rec", "start_t", "end_t", "pass_event", "aug_file", "aug_volume"])
        chunk_index = 0

    # load the audio files for data augmentation.
    if data_augmentation:
        aug_files = get_augmentation_files(augmentation_folder, samplerate, accepted_file_extensions)
        if len(aug_files) == 0: raise ValueError("Could not find any matching audio file for data augmentation.")

    # now load each recording and split it into chunks.
    for rec in t_labels["recording"].unique():
    
        print(f"Now processing recording -> {rec} ... ")

        # load the recording
        input_audio_file = input_folder / f"{rec}.wav"
        audio, sr = sf.read(input_audio_file)

        # check the sample rate
        assert sr == samplerate, (
            f"{input_audio_file.name} uses {sr} Hz instead of {samplerate}."
        )

        # calculate the start and end time of the first chunk. also convert times into sample indices.
        start_time = init_offset
        end_time = start_time + window_size
        start_sample = int(start_time * samplerate)
        end_sample = int(end_time * samplerate)

        # split the recording until the remaining audio is too short for the next chunk.
        while(end_sample <= len(audio)):
            
            chunk = audio[start_sample:end_sample]

            # data augmentation
            aug_file = ""
            if data_augmentation:
                aug_file = random.choice(list(aug_files.keys()))
                aug_audio = aug_files[aug_file]
                chunk = augment_audio(chunk, aug_audio, aug_volume)

            # save the chunk
            filename = output_folder / f"{chunk_index}.wav"
            sf.write(filename, chunk, samplerate, subtype="PCM_24")

            # ----------------------------------------------------------

            # now create the matching information and store it to the dataframe.
            
            # count the number of timestamps that mark a vehicle passing between the start and end time of this chunk.
            timestamps_current_chunk = t_labels[((t_labels.timestamp >= start_time) & (t_labels.timestamp < end_time) & (t_labels.recording == rec))]
            number_of_pass_events = len(timestamps_current_chunk)

            # if theres at least one vehicle in the chunk the pass_event is 1, otherwise 0.
            pass_event = 1 if number_of_pass_events >= 1 else 0

            # create the new row.
            c_labels.loc[len(c_labels)] = [
                chunk_index,
                rec,
                start_time,
                end_time,
                pass_event,
                aug_file,
                aug_volume
            ]

            # calculate the start and end time as well as the sample indices for the next chunk.
            start_time = start_time + window_size - overlap
            end_time = start_time + window_size
            start_sample = int(start_time * samplerate)
            end_sample = int(end_time * samplerate)

            # increase the chunk index
            chunk_index += 1

    # return the new dataframe and save it to the output folder als "chunk_labels.csv".
    c_labels.to_csv(output_folder / "chunk_labels.csv", sep=";")
    print(f"DONE. There are now {len(c_labels)} chunks.")
    return c_labels        