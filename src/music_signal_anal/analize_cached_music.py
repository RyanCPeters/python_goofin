from src import cache_path # Path object mapping to projects cache folder
from src import caching_paths # a dict of pathing data outside of project directory structure
from pathlib import Path
import librosa as lr
from librosa.display import waveplot
import numpy  as np
import winsound as ws



supported_music_types = {".mp3",".wav",".wave"}


def _get_track(track_path:str, track_name:str=None):
    track_path = cache_path.joinpath(track_path).resolve()
    if not track_path.is_absolute():
        _cache_path = Path(cache_path)
        while True:
            try:
                _track_path = _cache_path.relative_to(track_path)
                break
            except ValueError:
                if _cache_path.name:
                    _cache_path = _cache_path.parent
                else:
                    raise
        track_path = _track_path
    if track_name is None:
        for fname in track_path.iterdir():
            if fname.suffix.lower() in supported_music_types:
                track_name = fname.name
                break
        else:
            err = FileNotFoundError("No music tracks available in given directory")
            err.args += str(track_path),
            raise err
    return track_path.joinpath(track_name)

def analyze_track(track_path:str, track_name:str=None):
    """
    Track beats using time series input
    >>> y, sr = lr.load(lr.ex('choice'), duration=10)
    >>> tempo, beats = lr.beat.beat_track(y=y, sr=sr)
    >>> tempo
    135.99917763157896
    Print the frames corresponding to beats
    >>> beats
    array([  3,  21,  40,  59,  78,  96, 116, 135, 154, 173, 192, 211,
           230, 249, 268, 287, 306, 325, 344, 363])
    Or print them as timestamps
    >>> lr.frames_to_time(beats, sr=sr)
    array([0.07 , 0.488, 0.929, 1.37 , 1.811, 2.229, 2.694, 3.135,
           3.576, 4.017, 4.458, 4.899, 5.341, 5.782, 6.223, 6.664,
           7.105, 7.546, 7.988, 8.429])
    Track beats using a pre-computed onset envelope
    >>> onset_env = lr.onset.onset_strength(y, sr=sr,
    ...                                          aggregate=np.median)
    >>> tempo, beats = lr.beat.beat_track(onset_envelope=onset_env,
    ...                                        sr=sr)
    >>> tempo
    135.99917763157896
    >>> beats
    array([  3,  21,  40,  59,  78,  96, 116, 135, 154, 173, 192, 211,
           230, 249, 268, 287, 306, 325, 344, 363])
    Plot the beat events against the onset strength envelope
    >>> import matplotlib.pyplot as plt
    >>> hop_length = 512
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> times = lr.times_like(onset_env, sr=sr, hop_length=hop_length)
    >>> M = lr.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    >>> lr.display.specshow(lr.power_to_db(M, ref=np.max),
    ...                          y_axis='mel', x_axis='time', hop_length=hop_length,
    ...                          ax=ax[0])
    >>> ax[0].label_outer()
    >>> ax[0].set(title='Mel spectrogram')
    >>> ax[1].plot(times, lr.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r',
    ...            linestyle='-', label='Beats')
    >>> ax[1].legend()
    :param track_path:
    :type track_path:
    :param track_name:
    :type track_name:
    :return:
    :rtype:
    """
    tpath = _get_track(track_path,track_name)
    ws.PlaySound(str(tpath),ws.SND_ASYNC)
    input("Press enter to end")
    # audio_arr: np.ndarray
    # audio_arr,sr = lr.load(str(tpath))
    # # onset_env = lr.onset.onset_strength(audio_arr, sr=sr, aggregate=np.median)
    # # tempo, beats = lr.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # # hop_length = 512
    # # times = lr.times_like(onset_env, sr=sr, hop_length=hop_length)
    # # M = lr.feature.melspectrogram(y=audio_arr, sr=sr, hop_length=hop_length)
    # # p2db = lr.power_to_db(M, ref=np.max)
    # # fig = go.Figure()
    # # fig.add_scattergl(x=tuple(range(len(audio_arr))), y=audio_arr, mode="markers")
    # # fig.show()
    # # dbg_break = 0
    # onset_env = lr.onset.onset_strength(audio_arr, sr=sr, aggregate=np.median)
    # tempo, beats = lr.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # import matplotlib.pyplot as plt
    # hop_length = 512
    # fig1, ax1 = plt.subplots(nrows=2, sharex=True)
    # times = lr.times_like(onset_env, sr=sr, hop_length=hop_length)
    # M = lr.feature.melspectrogram(y=audio_arr, sr=sr, hop_length=hop_length)
    # lr.display.specshow(lr.power_to_db(M, ref=np.max), y_axis='mel', x_axis='time', hop_length=hop_length, ax=ax1[0])
    # ax1[0].label_outer()
    # ax1[0].set(title='Mel spectrogram')
    # ax1[1].plot(times, lr.util.normalize(onset_env),label='Onset strength')
    # ax1[1].vlines(times[beats], 0, 1, alpha=0.5, color='r',linestyle='-', label='Beats')
    # ax1[1].legend()
    # # plt.show()
    # #########
    # # using lr.decompose.decompose
    # S = np.abs(lr.stft(audio_arr))
    # # comps, acts = lr.decompose.decompose(S, n_components=8)
    # # # Sort components by ascending peak frequency
    # # comps, acts = lr.decompose.decompose(S, n_components=16, sort=True)
    # # Or with sparse dictionary learning
    # import sklearn.decomposition
    # T = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=16)
    # scomps, sacts = lr.decompose.decompose(S, transformer=T, sort=True)
    # fig2, ax2 = plt.subplots(nrows=1, ncols=2)
    # lr.display.specshow(lr.amplitude_to_db(scomps, ref=np.max), y_axis='log', ax=ax2[0])
    # ax2[0].set(title='Components')
    # lr.display.specshow(sacts, x_axis='time', ax=ax2[1])
    # ax2[1].set(ylabel='Components', title='Activations')
    # fig3, ax3 = plt.subplots(nrows=2, sharex=True, sharey=True)
    # lr.display.specshow(lr.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax3[0])
    # ax3[0].set(title='Input spectrogram')
    # ax3[0].label_outer()
    # S_approx = scomps.dot(sacts)
    # img = lr.display.specshow(lr.amplitude_to_db(S_approx, ref=np.max), y_axis='log', x_axis='time', ax=ax3[1])
    # ax3[1].set(title='Reconstructed spectrogram')
    # fig3.colorbar(img, ax=ax3, format="%+2.f dB")
    # plt.show()

if __name__ == '__main__':
    analyze_track(caching_paths["yt_music"])