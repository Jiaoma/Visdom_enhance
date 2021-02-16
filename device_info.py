import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    result=result.decode('utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def get_gpu_temperature_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=temperature.gpu',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    result = result.decode('utf-8')
    gpu_temperature = [int(x) for x in result.strip().split('\n')]
    gpu_temperature_map = dict(zip(range(len(gpu_temperature)), gpu_temperature))
    return gpu_temperature_map

def get_gpu_name():
    result = subprocess.check_output(
        [
            'nvidia-smi', '-L'
        ]
    )
    return result.decode('utf-8')

def get_nvidia_smi():
    result=subprocess.check_output([
        'nvidia-smi'
    ])
    return result.decode('utf-8')

def get_disk_usage():
    result=subprocess.check_output([
        'df','-h','/etc'
    ])
    return result.decode('utf-8')

if __name__=='__main__':
    import visdom
    vis=visdom.Visdom()

    vis.text(get_gpu_name().replace('\n','<br>'))
    vis.text(get_nvidia_smi().replace('\n','<br>'))
    vis.text(get_disk_usage().replace('\n','<br>'))