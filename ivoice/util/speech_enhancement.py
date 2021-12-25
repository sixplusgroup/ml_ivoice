import os


def DeepXi(input_dir, output_dir, version="mhanet-1.1c", gain="mmse-lsa"):
    """

    Args:
        input_dir: string
        output_dir: string
        version: which model is used to infer
            default: 'mhanet-1.1c'
            option: 'mhanet-1.1c', 'mhanet-1.0c', 'resnet-1.1c', 'resnet-1.1n',
                    'rdlnet-1.0n', 'resnet-1.0c', 'resnet-1.0n', 'reslstm-1.0c'
        gain: gain function
            default: 'mmse-lsa'
            option:
                'ibm' - ideal binary mask (IBM),
                'wf' - Wiener filter (WF),
                'srwf' - square-root Wiener filter (SRWF),
                'cwf' - constrained Wiener filter (cWF),
                'mmse-stsa' - minimum-mean square error short-time spectral smplitude (MMSE-STSA) estimator,
                'mmse-lsa' - minimum-mean square error log-spectral amplitude (MMSE-LSA) estimator.

    Returns:

    """
    VIRTUAL_ENV = os.path.abspath("../approach/speech_enhancement/DeepXi/venv")
    os.environ['PATH'] = VIRTUAL_ENV + "/bin:" + os.environ['PATH']
    os.chdir('../approach/speech_enhancement/DeepXi')
    os.system('./run.sh VER=%s INFER=1 GAIN=%s INPUT_DIR=%s OUTPUT_DIR=%s' % (version, gain, input_dir, output_dir))


DeepXi(
    input_dir=os.path.abspath("../data/DeepXi/input"),
    output_dir=os.path.abspath("../data/DeepXi/output"),
    version="mhanet-1.1c",
    gain="mmse-lsa"
)
