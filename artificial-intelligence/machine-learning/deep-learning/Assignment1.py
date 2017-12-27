# These are all the modules we'll be using later.
# Make sure you can import them before proceeding further.
import os
import sys
import tarfile

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve

# Download data
url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = './data'  # Change me to store data elsewhere


def download_progress_hook(count, block_size, total_size):
    """A hook to report the progress of a download.
    This is mostly intended for users with slow internet connections.
    Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * block_size * 100 / total_size)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

    last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    destination_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(destination_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, destination_filename,
                                  reporthook=download_progress_hook)
        print('\nDownload Complete!')
    download_stat_info = os.stat(destination_filename)
    if download_stat_info.st_size == expected_bytes:
        print('Found and verified', destination_filename)
    else:
        raise Exception(
            'Failed to verify {0}. Can you get to it with a browser?'
            .format(destination_filename))
    return destination_filename


train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

# Extract downloaded files
num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('{} already present - Skipping extraction of {}.'
              .format(root, filename))
    else:
        print('Extracting data for {}. This may take a while. Please wait.'
              .format(root))
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected {:d} folders, one per class. Found {:d} instead.'
            .format(num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


# Show sample of downloaded files
def show_images_horizontally(list_of_files):
    fig = plt.figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        fig.add_subplot(1, number_of_files, i + 1)
        image = plt.imread(list_of_files[i])
        plt.imshow(image, cmap='Greys_r')
        plt.axis('off')


def show_image_sample(data_folders):
    class_files = []
    for data_folder in data_folders:
        data_folder_full_path = data_folder
        random_index = np.random.randint(0, len(
            os.listdir(data_folder_full_path)) - 1)
        class_files.append(
            os.path.join(data_folder_full_path,
                         os.listdir(data_folder_full_path)[random_index]))
    show_images_horizontally(class_files)


show_image_sample(train_folders)
show_image_sample(test_folders)

# Load files in pickles
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = \
                (ndimage.imread(image_file).astype(float) - pixel_depth / 2) \
                / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception(
                    'Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as exc:
            print('Could not read:', image_file, ':', exc,
                  '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: {:d} < {:d}'
                        .format(num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as set_file:
                    pickle.dump(dataset, set_file, pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                print('Unable to save data to', set_filename, ':', exc)

    return dataset_names


train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


# Show sample of data from pickles
def show_data_horizontally(list_of_data):
    fig = plt.figure()
    number_of_data = len(list_of_data)
    for i in range(number_of_data):
        fig.add_subplot(1, number_of_data, i + 1)
        plt.imshow(list_of_data[i], cmap='Greys_r')
        plt.axis('off')
        plt.title(i)


def show_data_sample(list_of_datasets):
    class_data = []
    for dataset in list_of_datasets:
        with open(dataset, 'rb') as file:
            x = pickle.load(file)
        random_index = np.random.randint(0, len(x) - 1)
        class_data.append(x[random_index])
    show_data_horizontally(class_data)


show_data_sample(train_datasets)
show_data_sample(test_datasets)


# %% Verify balancing in pickles
def verify_balancing(datasets):
    pass


verify_balancing(train_datasets)
verify_balancing(test_datasets)


# Merge datasets with labels
def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, merge_train_size, merge_valid_size=0):
    merge_num_classes = len(pickle_files)
    merge_valid_dataset, merge_valid_labels = \
        make_arrays(merge_valid_size, image_size)
    merge_train_dataset, merge_train_labels = \
        make_arrays(merge_train_size, image_size)
    valid_size_per_class = merge_valid_size // merge_num_classes
    train_size_per_class = merge_train_size // merge_num_classes

    start_v, start_t = 0, 0
    end_v, end_t = valid_size_per_class, train_size_per_class
    end_l = valid_size_per_class + train_size_per_class
    for label, current_pickle_file in enumerate(pickle_files):
        try:
            with open(current_pickle_file, 'rb') as file:
                letter_set = pickle.load(file)
                # let's shuffle the letters to have random validation
                # and training set
                np.random.shuffle(letter_set)
                if merge_valid_dataset is not None:
                    valid_letter = letter_set[:valid_size_per_class, :, :]
                    merge_valid_dataset[start_v:end_v, :, :] = valid_letter
                    merge_valid_labels[start_v:end_v] = label
                    start_v += valid_size_per_class
                    end_v += valid_size_per_class

                train_letter = letter_set[valid_size_per_class:end_l, :, :]
                merge_train_dataset[start_t:end_t, :, :] = train_letter
                merge_train_labels[start_t:end_t] = label
                start_t += train_size_per_class
                end_t += train_size_per_class
        except Exception as e:
            print('Unable to process data from', current_pickle_file, ':', e)
            raise

    return merge_valid_dataset, \
        merge_valid_labels, \
        merge_train_dataset, \
        merge_train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# Randomize datasets
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


# Show data after randomization
def show_data_with_labels_horizontally(list_of_labels_with_data):
    fig = plt.figure()
    number_of_data = len(list_of_labels_with_data)
    for i in range(number_of_data):
        fig.add_subplot(1, number_of_data, i + 1)
        plt.imshow(list_of_labels_with_data[i][1], cmap='Greys_r')
        plt.axis('off')
        plt.title(list_of_labels_with_data[i][0])


def show_data_with_labels_sample(dataset, labels):
    class_data = []
    for i in range(num_classes):
        random_index = np.random.randint(0, len(dataset) - 1)
        class_data.append([labels[random_index], dataset[random_index]])
    show_data_with_labels_horizontally(class_data)


show_data_with_labels_sample(train_dataset, train_labels)
show_data_with_labels_sample(test_dataset, test_labels)

# Save generated datasets

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
    with open(pickle_file, 'wb') as f:
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

stat_info = os.stat(pickle_file)
print('Compressed pickle size:', stat_info.st_size)

# Create sanitized datasets (without duplicates)
# TODO
# pickle_file_sanitized = os.path.join(data_root, 'notMNIST_sanitized.pickle')
#
# try:
#     with open(pickle_file_sanitized, 'wb') as f:
#         save = {
#             'train_dataset': train_dataset_sanitized,
#             'train_labels': train_labels_sanitized,
#             'valid_dataset': valid_dataset_sanitized,
#             'valid_labels': valid_labels_sanitized,
#             'test_dataset': test_dataset_sanitized,
#             'test_labels': test_labels_sanitized,
#         }
#         pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
# except Exception as e:
#     print('Unable to save data to', pickle_file, ':', e)
#     raise
#
# stat_info = os.stat(pickle_file)
# print('Compressed pickle size sanitized:', stat_info.st_size)
