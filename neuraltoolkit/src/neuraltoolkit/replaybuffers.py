import numpy as np
from collections import deque


class Replay:
    def __init__(
            self, 
            state=None, 
            action=None, 
            reward=None, 
            new_state=None, 
            done=False, 
            td_error=0
            ):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.done = done
        self.td_error = td_error


class MultiReplayBuffer:
    def __init__(
            self, 
            buffers=1,
            buffer_split=None,
            inputDim=None, 
            capacity=None, 
            min=None, 
            n_step_cap=1, 
            discount_rate=0.99, 
            alpha=0.6
            ):
        if buffer_split == None:
            self.buffer_split = [1 / buffers] * buffers
        self.buffer_count = buffers
        self.capacity = capacity
        self.min = min
        self.alpha = alpha

        self.args = locals().copy()
        del self.args["self"]
        self.buffers = [ReplayBuffer(**self.args) for i in range(self.buffer_count)]

        self.inputDim = inputDim

    def add_buffer(self, buffer_split=None):
        self.buffer_count += 1
        if buffer_split == None:
            self.buffer_split = [1 / self.buffer_count] * self.buffer_count
        else:
            if (len(buffer_split) != self.buffer_count or max(buffer_split) > 1 or min(buffer_split) < 0):
                raise ValueError("buffer_split argument either does not match number of buffers or elements are not add up to one.")
            self.buffer_split = buffer_split
        self.buffers.append(ReplayBuffer(**self.args))

    def get_sizes(self):
        sizes = [buffer._size for buffer in self.buffers]
        return tuple(sizes)
    
    def size(self):
        sizes = np.array(self.get_sizes())
        return sizes.sum()
    
    def calc_final_n_steps(self, buffer=-1):
        self.buffers[buffer].calc_final_n_steps()

    def add_replay(self, state=None, action=None, reward=None, new_state=None, done=False, td_error=0, buffer=-1):
        args = locals().copy()
        del args["self"]
        self.buffers[buffer].add_replay(**args)

    def add_replay(self, replay:Replay=None, buffer=-1):
        args = locals().copy()
        del args["self"]
        self.buffers[buffer].add_replay(**args)

    def sample(self, batch_size, beta):
        batch_splits = [
            int(batch_size * self.buffer_split[i]) for i in range(self.buffer_count)
        ]
        New_batch_size = np.sum(batch_splits)

        states = [np.zeros((New_batch_size, *shape)) for shape in self.inputDim]
        new_states = [np.zeros((New_batch_size, *shape)) for shape in self.inputDim]
        actions = np.zeros((New_batch_size,), dtype=int)
        rewards = np.zeros((New_batch_size,))
        done_flags = np.empty(New_batch_size, dtype=object)

        indices = np.zeros((New_batch_size, 2))
        weights = np.zeros((New_batch_size))
        pos = 0
        buffer_idx = 0
        for buffer, batch_split in zip(self.buffers, batch_splits):
            _samples, _indices, _weights = buffer.sample(batch_split, beta)

            indices[pos:pos + batch_split, 0] = buffer_idx
            indices[pos:pos + batch_split, 1] = _indices
            weights[pos:pos + batch_split] = _weights

            for i in range(len(self.inputDim)):
                states[i][pos:pos + batch_split] = _samples["states"][i]
                new_states[i][pos:pos + batch_split] = _samples["new_states"][i]
            actions[pos:pos + batch_split] = _samples["actions"]
            rewards[pos:pos + batch_split] = _samples["rewards"]
            done_flags[pos:pos + batch_split] = _samples["done_flags"]

            pos += batch_split
            buffer_idx += 1

        samples = {
            "states":states,
            "new_states":new_states,
            "actions":actions,
            "rewards":rewards,
            "done_flags":done_flags,
        }

        return samples, indices, weights
    
    def update_td_errors(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            i, j = np.astype(idx, int)
            self.buffers[i].priorities[j] = abs(td_error) + 1e-6


class ReplayBuffer:
    def __init__(
            self, 
            inputDim, 
            capacity, 
            min, 
            n_step_cap=1, 
            discount_rate=0.99, 
            alpha=0.6, 
            **kwargs):
        self.capacity = capacity
        self.min = min
        self.pos = 0
        self._size = 0

        self.n_step = deque(maxlen=n_step_cap)
        self.discount_rate = discount_rate

        self.alpha = alpha

        self.states = [
            np.zeros((capacity, *shape)) for shape in inputDim
        ]
        self.new_states = [
            np.zeros((capacity, *shape)) for shape in inputDim
        ]
        self.actions = np.zeros((capacity,), dtype=int)
        self.rewards = np.zeros((capacity,))
        self.done_flags = np.empty(capacity, dtype=object)
        self.td_errors = np.zeros((capacity,))
        self.priorities = np.zeros((capacity,))

    def size(self):
        return self._size

    def wrap_inputs(self, replay:Replay):
        if not isinstance(replay.state, (tuple, list)):
            replay.state = (replay.state,)

        if not isinstance(replay.new_state, (tuple, list)):
            replay.new_state = (replay.new_state,)
    
    def calc_final_n_steps(self):
        for i in range(len(self.n_step)):
            self.push_replay()
    
    def calc_n_step_reward(self):
        reward = 0
        for i, step in enumerate(self.n_step):
            reward += step.reward * self.discount_rate ** i
        return reward

    def push_replay(self):
        for i in range(len(self.states)):
            self.states[i][self.pos] = self.n_step[0].state[i]
        for i in range(len(self.new_states)):
            self.new_states[i][self.pos] = self.n_step[-1].new_state[i]
        self.actions[self.pos] = self.n_step[0].action
        self.rewards[self.pos] = self.calc_n_step_reward()
        self.done_flags[self.pos] = self.n_step[0].done
        self.td_errors[self.pos] = self.n_step[0].td_error
        self.priorities[self.pos] = self.priorities.max() if self._size > 0 else 1.0

        self.pos = (self.pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self.n_step.popleft()


    def add_replay(self, state=None, action=None, reward=None, new_state=None, done=False, td_error=0, **kwags):
        replay = Replay(state, action, reward, new_state, done, td_error)
        self.wrap_inputs(replay)
        self.n_step.append(replay)
        if len(self.n_step) == self.n_step.maxlen:
            self.push_replay()

    def add_replay(self, replay:Replay=None, **kwargs):
        if not isinstance(replay, Replay):
            raise ValueError(f"expected {Replay} but got {replay}")
        
        self.wrap_inputs(replay)
        self.n_step.append(replay)
        if len(self.n_step) == self.n_step.maxlen:
            self.push_replay()

    def sample(self, batch_size, beta):
        if self._size == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self._size, batch_size, p=probs)

        total = self._size
        weights = (total * probs[indices]) ** -beta
        weights /= weights.max()

        states = []
        new_states = []
        for i in range(len(self.states)):
            states.append(np.array([self.states[i][idx] for idx in indices]))
            new_states.append(np.array([self.new_states[i][idx] for idx in indices]))
        samples = {
            "states": states,
            "new_states": new_states,
            "actions": np.array([self.actions[idx] for idx in indices]),
            "rewards": np.array([self.rewards[idx] for idx in indices]),
            "done_flags": np.array([self.done_flags[idx] for idx in indices]),
        }

        return samples, indices, weights
    
    def update_td_errors(self, indices, td_errors):
        for index, td_error in zip(indices, td_errors):
            self.priorities[index] = abs(td_error) + 1e-6