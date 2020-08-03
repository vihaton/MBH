import os


def add_trailing_slash(dir_name):
    dir_name += '/' if dir_name[-1] != '/' else ''
    return dir_name


def get_folders(root_dir):
    dirs = None
    for (path, dirs, files) in os.walk(root_dir):
        break
    return dirs


def get_files_in_dir(sdicts_dir, verbose=False):
    for (path, dirs, files) in os.walk(sdicts_dir):
        if verbose:
            print('path: ', path)
            print('dirs', dirs)
            print('files')
            for i, file in enumerate(files):
                print('\t', i, file)
        break
    return files


def get_pt_files_in_dir(sdicts_dir):
    pt_files = []
    for (path, dirs, files) in os.walk(sdicts_dir):
        print('path: ', path)
        print('dirs', dirs)
        print('files')
        for i, file in enumerate(files):
            print('\t', i, file)
            name, ending = file.split('.')
            if ending == 'pt':
                pt_files.append(file)
        break
    return pt_files
