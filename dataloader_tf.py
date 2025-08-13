# import tensorflow as tf
# import h5py
# import numpy as np

# class BranchTraceDatasetTF:
#     def __init__(self, trace_paths, br_pc, history_lengths, pc_bits, pc_hash_bits, hash_dir_with_pc):
#         self.trace_paths = trace_paths
#         self.br_pc = br_pc
#         self.history_lengths = history_lengths  # [42, 78, 150, 294, 582]
#         self.max_history = max(history_lengths)
#         self.pc_bits = pc_bits
#         self.pc_hash_bits = pc_hash_bits
#         self.hash_dir_with_pc = hash_dir_with_pc
#         self.samples = self.load_all_indices()

#     def load_all_indices(self):
#         samples = []
#         for path in self.trace_paths:
#             with h5py.File(path, 'r') as f:
#                 history = f['history'][:]
#                 br_indices_ds = f[f'br_indices_{self.br_pc}'][:]
#                 br_indices_ds = br_indices_ds[br_indices_ds >= self.max_history]
#                 for idx in br_indices_ds:
#                     samples.append((path, idx))
#         return samples

#     def __len__(self):
#         return len(self.samples)

#     def preprocess_history(self, history):
#         pc_bits, pc_hash_bits, hash_dir_with_pc = self.pc_bits, self.pc_hash_bits, self.hash_dir_with_pc
#         pc_mask = (1 << (1 + pc_bits)) - 1
#         history = np.bitwise_and(history, pc_mask)
#         if hash_dir_with_pc:
#             if pc_hash_bits < (pc_bits + 1):
#                 unprocessed_bits = pc_bits + 1 - pc_hash_bits
#                 pc_hash_mask = ((1 << pc_hash_bits) - 1)
#                 shift_count = 1
#                 temp = np.empty_like(history)
#                 while unprocessed_bits > 0:
#                     np.right_shift(history, shift_count * pc_hash_bits, out=temp)
#                     np.bitwise_and(temp, pc_hash_mask, out=temp)
#                     np.bitwise_xor(history, temp, out=history)
#                     shift_count += 1
#                     unprocessed_bits -= pc_hash_bits
#                 np.bitwise_and(history, pc_hash_mask, out=history)
#         else:
#             if pc_hash_bits < pc_bits:
#                 unprocessed_bits = pc_bits - pc_hash_bits
#                 pc_hash_mask = ((1 << pc_hash_bits) - 1) << 1
#                 shift_count = 1
#                 temp = np.empty_like(history)
#                 while unprocessed_bits > 0:
#                     np.right_shift(history, shift_count * pc_hash_bits, out=temp)
#                     np.bitwise_and(temp, pc_hash_mask, out=temp)
#                     np.bitwise_xor(history, temp, out=history)
#                     shift_count += 1
#                     unprocessed_bits -= pc_hash_bits
#                 stew_mask = (1 << (pc_hash_bits + 1)) - 1
#                 np.bitwise_and(history, stew_mask, out=history)
#         return history.astype(np.int32)

#     def generator(self):
#         for path, idx in self.samples:
#             with h5py.File(path, 'r') as f:
#                 history_chunk = f['history'][idx - self.max_history: idx + 1]
#                 history_chunk = self.preprocess_history(history_chunk)
#                 # Extract slices
#                 slices = []
#                 for l in self.history_lengths:
#                     slices.append(history_chunk[-l:])
#                 label = 1.0 if history_chunk[-1] & 1 else 0.0
#                 yield tuple(slices), label

#     def get_dataset(self, batch_size=128, shuffle=True):
#         output_signature = (
#             tuple(tf.TensorSpec(shape=(l,), dtype=tf.int32) for l in self.history_lengths),
#             tf.TensorSpec(shape=(), dtype=tf.float32)
#         )
#         dataset = tf.data.Dataset.from_generator(
#             self.generator, output_signature=output_signature
#         )
#         if shuffle:
#             dataset = dataset.shuffle(buffer_size=1024)
#         dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#         return dataset


import tensorflow as tf
import h5py
import numpy as np

# class BranchTraceDatasetTF:
#     def __init__(self, trace_paths, br_pc, history_lengths, pc_bits, pc_hash_bits, hash_dir_with_pc):
#         self.trace_paths = trace_paths
#         self.br_pc = br_pc
#         self.history_lengths = history_lengths  # This is now a list
#         self.max_history = max(history_lengths)  # Use the max history length
#         self.pc_bits = pc_bits
#         self.pc_hash_bits = pc_hash_bits
#         self.hash_dir_with_pc = hash_dir_with_pc
#         self.samples = self.load_all_indices()

#     def load_all_indices(self):
#         samples = []
#         for path in self.trace_paths:
#             with h5py.File(path, 'r') as f:
#                 history = f['history'][:]
#                 br_indices_ds = f[f'br_indices_{self.br_pc}'][:]
#                 br_indices_ds = br_indices_ds[br_indices_ds >= self.max_history]
#                 for idx in br_indices_ds:
#                     samples.append((path, idx))
#         return samples

#     def __len__(self):
#         return len(self.samples)

#     def preprocess_history(self, history):
#         pc_bits, pc_hash_bits, hash_dir_with_pc = self.pc_bits, self.pc_hash_bits, self.hash_dir_with_pc
#         pc_mask = (1 << (1 + pc_bits)) - 1
#         history = np.bitwise_and(history, pc_mask)
#         if hash_dir_with_pc:
#             if pc_hash_bits < (pc_bits + 1):
#                 unprocessed_bits = pc_bits + 1 - pc_hash_bits
#                 pc_hash_mask = ((1 << pc_hash_bits) - 1)
#                 shift_count = 1
#                 temp = np.empty_like(history)
#                 while unprocessed_bits > 0:
#                     np.right_shift(history, shift_count * pc_hash_bits, out=temp)
#                     np.bitwise_and(temp, pc_hash_mask, out=temp)
#                     np.bitwise_xor(history, temp, out=history)
#                     shift_count += 1
#                     unprocessed_bits -= pc_hash_bits
#                 np.bitwise_and(history, pc_hash_mask, out=history)
#         else:
#             if pc_hash_bits < pc_bits:
#                 unprocessed_bits = pc_bits - pc_hash_bits
#                 pc_hash_mask = ((1 << pc_hash_bits) - 1) << 1
#                 shift_count = 1
#                 temp = np.empty_like(history)
#                 while unprocessed_bits > 0:
#                     np.right_shift(history, shift_count * pc_hash_bits, out=temp)
#                     np.bitwise_and(temp, pc_hash_mask, out=temp)
#                     np.bitwise_xor(history, temp, out=history)
#                     shift_count += 1
#                     unprocessed_bits -= pc_hash_bits
#                 stew_mask = (1 << (pc_hash_bits + 1)) - 1
#                 np.bitwise_and(history, stew_mask, out=history)
#         return history.astype(np.int32)

#     def generator(self):
#         for path, idx in self.samples:
#             with h5py.File(path, 'r') as f:
#                 history_chunk = f['history'][idx - self.max_history: idx + 1]
#                 history_chunk = self.preprocess_history(history_chunk)
                
#                 # Extract slices based on history_lengths
#                 slices = [history_chunk[-length:] for length in self.history_lengths]
#                 label = 1.0 if history_chunk[-1] & 1 else 0.0
#                 yield tuple(slices), label

#     def get_dataset(self, batch_size=128, shuffle=True):
#         output_signature = (
#             tuple(tf.TensorSpec(shape=(length,), dtype=tf.int32) for length in self.history_lengths),
#             tf.TensorSpec(shape=(), dtype=tf.float32)
#         )
#         dataset = tf.data.Dataset.from_generator(
#             self.generator, output_signature=output_signature
#         )
#         if shuffle:
#             dataset = dataset.shuffle(buffer_size=1024)
#         dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#         return dataset

class BranchTraceDatasetTFSingle:
    def __init__(self, trace_paths, br_pc, history_length, pc_bits, pc_hash_bits, hash_dir_with_pc):
        self.trace_paths = trace_paths
        self.br_pc = br_pc
        self.history_length = int(history_length)
        self.pc_bits = pc_bits
        self.pc_hash_bits = pc_hash_bits
        self.hash_dir_with_pc = hash_dir_with_pc
        self.samples = self._load_all_indices()

    def _load_all_indices(self):
        samples = []
        for path in self.trace_paths:
            with h5py.File(path, 'r') as f:
                br_idx = f[f'br_indices_{self.br_pc}'][:]
                br_idx = br_idx[br_idx >= self.history_length]  # ensure enough history
                for idx in br_idx:
                    samples.append((path, int(idx)))
        return samples

    def __len__(self):
        return len(self.samples)

    # ---- same hashing/bit-mixing as your original ----
    def _preprocess_history(self, history):
        pc_bits, pc_hash_bits, hash_dir_with_pc = self.pc_bits, self.pc_hash_bits, self.hash_dir_with_pc
        pc_mask = (1 << (1 + pc_bits)) - 1
        history = np.bitwise_and(history, pc_mask)

        if hash_dir_with_pc:
            if pc_hash_bits < (pc_bits + 1):
                unprocessed_bits = pc_bits + 1 - pc_hash_bits
                pc_hash_mask = ((1 << pc_hash_bits) - 1)
                shift_count = 1
                temp = np.empty_like(history)
                while unprocessed_bits > 0:
                    np.right_shift(history, shift_count * pc_hash_bits, out=temp)
                    np.bitwise_and(temp, pc_hash_mask, out=temp)
                    np.bitwise_xor(history, temp, out=history)
                    shift_count += 1
                    unprocessed_bits -= pc_hash_bits
                np.bitwise_and(history, pc_hash_mask, out=history)
        else:
            if pc_hash_bits < pc_bits:
                unprocessed_bits = pc_bits - pc_hash_bits
                pc_hash_mask = ((1 << pc_hash_bits) - 1) << 1
                shift_count = 1
                temp = np.empty_like(history)
                while unprocessed_bits > 0:
                    np.right_shift(history, shift_count * pc_hash_bits, out=temp)
                    np.bitwise_and(temp, pc_hash_mask, out=temp)
                    np.bitwise_xor(history, temp, out=history)
                    shift_count += 1
                    unprocessed_bits -= pc_hash_bits
                stew_mask = (1 << (pc_hash_bits + 1)) - 1
                np.bitwise_and(history, stew_mask, out=history)

        return history.astype(np.int32)

    def generator(self):
        # yields: (seq[int32, history_length], label[float32])
        for path, idx in self.samples:
            with h5py.File(path, 'r') as f:
                # include current element so we can read label from the last item
                chunk = f['history'][idx - self.history_length : idx + 1]
                chunk = self._preprocess_history(chunk)

                x = chunk[-self.history_length:]           # model input
                y = 1.0 if (chunk[-1] & 1) else 0.0        # label from LSB of last item
                yield x, np.float32(y)

    def get_dataset(self, batch_size=128, shuffle=True, repeat=False):
        output_signature = (
            tf.TensorSpec(shape=(self.history_length,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(self.generator, output_signature=output_signature)
        if shuffle:
            ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
        if repeat:
            ds = ds.repeat()
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
