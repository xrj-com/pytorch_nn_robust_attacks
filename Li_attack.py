import sys
import torch
from torch.autograd import Variable
import numpy as np

DECREASE_FACTOR = 0.9   # 0<f<1, rate at which we shrink tau; larger is more accurate
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # abort gradient descent upon first valid solution
INITIAL_CONST = 1e-5    # the first value of c to start at
LEARNING_RATE = 5e-3    # larger values converge faster to less accurate results
LARGEST_CONST = 2e+1    # the largest value of c to go up to before giving up
REDUCE_CONST = False    # try to lower c each iteration; faster to set to false
TARGETED = True         # should we target one specific class? or just be wrong?
CONST_FACTOR = 2.0      # f>1, rate at which we increase constant, smaller better


class CarliniLi:
    def __init__(self,  model, shape,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, largest_const=LARGEST_CONST,
                 reduce_const=REDUCE_CONST, decrease_factor=DECREASE_FACTOR,
                 const_factor=CONST_FACTOR, is_cuda=True):
        """
        The L_infinity optimized attack. 

        Returns adversarial examples for the supplied model.

        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. Should be set to a very small
          value (but positive).
        largest_const: The largest constant to use until we report failure. Should
          be set to a very large value.
        reduce_const: If true, after each successful attack, make const smaller.
        decrease_factor: Rate at which we should decrease tau, less than one.
          Larger produces better quality results.
        const_factor: The rate at which we should increase the constant, when the
          previous constant failed. Should be greater than one, smaller is better.
        """
        self.model = model
        self.shape = shape
        self.is_cuda = is_cuda

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.DECREASE_FACTOR = decrease_factor
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor

        self.grad = self.gradient_descent(model, self.shape)

    def gradient_descent(self, model, shape):
        def compare(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y

        def doit(oimgs, labs, starts, tt, CONST):
            # convert to tanh-space
            imgs = np.arctanh(np.array(oimgs) * 1.999999)
            starts = np.arctanh(np.array(starts) * 1.999999)

            # initialize the variables
            if self.is_cuda:
                modifier = Variable(torch.zeros(shape).cuda(), requires_grad=True)

                timg = Variable(torch.from_numpy(imgs).cuda(), requires_grad=True)
                tlab = Variable(torch.FloatTensor(labs).cuda(), requires_grad=True)
                simg = Variable(torch.from_numpy(starts).cuda(), requires_grad=True)
                zero = Variable(torch.zeros(1).cuda(), requires_grad=True)
            else:
                modifier = Variable(torch.zeros(shape), requires_grad=True)

                timg = Variable(torch.from_numpy(imgs), requires_grad=True)
                tlab = Variable(torch.FloatTensor(labs), requires_grad=True)
                simg = Variable(torch.from_numpy(starts), requires_grad=True)
                zero = Variable(torch.zeros(1), requires_grad=True)

            tau = tt
            const = CONST

            optimizer = torch.optim.Adam(
                        [{'params': modifier}], lr=self.LEARNING_RATE)

            while CONST < self.LARGEST_CONST:
                # try solving for each value of the constant
                print('try const', CONST)

                for step in range(self.MAX_ITERATIONS):
                    optimizer.zero_grad()
                    newimg = torch.tanh(modifier + simg) / 2.0

                    output = model(newimg)
                    orig_output = model(torch.tanh(timg) / 2)

                    # print(output)
                    # print(orig_output)

                    real = torch.sum((tlab) * output)
                    other = torch.max((1 - tlab) * output - (tlab * 10000))
                    
    
                    if self.TARGETED:
                        # if targetted, optimize for making the other class most likely
                        
                        loss1 = torch.max(other - real, zero)
                    else:
                        # if untargeted, optimize for making this class least likely.
                        
                        loss1 = torch.max(real - other, zero)

                    loss2 = torch.sum(torch.max(zero, torch.abs(newimg - torch.tanh(timg) / 2) - tau))
                    loss = const * loss1 + loss2

                    if step % (self.MAX_ITERATIONS // 10) == 0:
                        print('step:{} loss:{} loss1:{} loss2:{}'.format(step, 
                        loss.cpu().data[0], loss1.cpu().data[0], loss2.cpu().data[0]))

                    # perform the update step
                    
                    loss.backward()
                    optimizer.step()

                    works = loss.cpu().data[0]
                    # it worked
                    if works < .0001 * CONST and (self.ABORT_EARLY or step == CONST - 1):
                        get = output.cpu().data.numpy()
                        works = compare(np.argmax(get), np.argmax(labs))
                        if works:
                            scores = output.cpu().data.numpy()
                            origscores = orig_output.cpu().data.numpy()
                            nimg = newimg.cpu().data.numpy()
                            l2s = np.square(
                                nimg - np.tanh(imgs) / 2).sum(axis=(1, 2, 3))

                            return scores, origscores, nimg, CONST

                # we didn't succeed, increase constant and try again
                CONST *= self.const_factor

        return doit

    def attack(self, imgs, targets):
        """
        Perform the L_0 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        imgs = np.transpose(imgs, [0, 3, 1, 2])
        for img, target in zip(imgs, targets):
            r.extend(self.attack_single(img, target))
        
        return np.transpose(np.array(r), [0, 2, 3, 1])

    def attack_single(self, img, target):
        """
        Run the attack on a single image and label
        """

        # the previous image

        prev = np.copy(img).reshape(self.shape)
        tau = 1.0
        const = self.INITIAL_CONST

        while tau > 1. / 256:
            # try to solve given this tau value
            res = self.grad([np.copy(img)], [target],
                            np.copy(prev), tau, const)
            if res == None:
                # the attack failed, we return this as our final answer
                return prev

            scores, origscores, nimg, const = res
            if self.REDUCE_CONST:
                const /= 2

            # the attack succeeded, reduce tau and try again

            actualtau = np.max(np.abs(nimg - img))

            if actualtau < tau:
                tau = actualtau

            print("Tau", tau)

            prev = nimg
            tau *= self.DECREASE_FACTOR
        return prev
