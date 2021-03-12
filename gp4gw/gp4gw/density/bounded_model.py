import tensorflow as tf
import tensorflow_probability as tfp

float_type = tf.float64

class BoundedDensityModel:

    def __init__(self,
                 likelihood,
                 param_names,
                 priors,
                 parameters_scaler,
                 y_scaler,
                 ):
        assert list(param_names), '{} is not of type list'.format(param_names)

        self.likelihood = likelihood
        self.param_names = param_names
        self.parameters_scaler = parameters_scaler
        self.y_scaler = y_scaler
        self.priors = priors
        if any(key == 'dec' for key in param_names):
            self.bijector_list = [tfp.bijectors.Softplus()] * (len(param_names) - 1)
            self.bijector_list.insert(param_names.index('dec'), tfp.bijectors.Identity())
        else:
            self.bijector_list = [tfp.bijectors.Softplus()] * len(param_names)

    def log_prior(self, x_scaled):
        """
        Function to evaluate the log prior.
        This is a piece-wise function returning either 0 or -inf
        """
        x = scale_inverse_transform(self.parameters_scaler, x_scaled)
        ln_prior = {}
        for key in self.param_names:
            try:
                ln_prior[key] = self.priors[key].log_prob(x[:, self.param_names.index(key)])
            except TypeError:
                x = tf.cast(x, tf.float32)
                ln_prior[key] = self.priors[key].log_prob(x[:, self.param_names.index(key)])
        ln_prior_sum = tf.math.reduce_sum(tf.convert_to_tensor(list(ln_prior.values()), dtype=float_type),
                                          axis=0)
        # If prior is finite, replace with zero
        zero_prior = tf.zeros_like(ln_prior_sum)
        constraints = tf.math.is_inf(ln_prior_sum)
        tensor_log_prior = tf.where(constraints, ln_prior_sum, zero_prior)
        return tensor_log_prior


    def log_prob(self, x_scaled):
        """
        Function to evaluate the log prob.
        This is the GP log mean + log prior.
        """
        predictions = self.likelihood(x_scaled)
        # this function transform the gp mean back to "real" space
        mean_with_negatives = scale_inverse_transform(self.y_scaler, predictions[0])
        # this function replaces negative numbers with zero
        mean = tf.nn.relu(mean_with_negatives)
        log_mean = tf.math.log(mean)
        tensor_log_prior = self.log_prior(x_scaled)
        prob = tf.reshape(log_mean, shape=tensor_log_prior.shape) + tensor_log_prior
        return prob

    def mult_var_by_prior(self, x_scaled):
        """
        Function to evaluate GP variance with priors
        """
        model_var = self.likelihood(x_scaled)[1]
        tensor_log_prior = self.log_prior(x_scaled)
        return tf.reshape(model_var, shape=tensor_log_prior.shape) * tf.math.exp(tensor_log_prior)

    def predict(self, x_scaled):
        """
        Returns GP mean and variance in "scaled" space
        (same as gpflow model.predict_f with bounds applied)
        """
        model_var_with_prior = self.mult_var_by_prior(x_scaled)
        scaled_space_mean = self.y_scaler.transform(tf.reshape(tf.math.exp(self.log_prob(x_scaled)), [-1, 1]))
        return scaled_space_mean, tf.reshape(model_var_with_prior, [-1, 1]).numpy()
    
    def predict_samples(self, x_scaled):
        """
        Returns multiple GP samples
        """
        all_predictions = []
        predictions = self.likelihood(x_scaled)
        #Todo: this loop can easily be vectorised
        # fix this when plots have been made
        for i in range(predictions.shape[0]):
            mean_with_negatives = scale_inverse_transform(self.y_scaler,
                                                          predictions[i])
            mean = self.remove_random_gp_jitters(mean_with_negatives)
            log_mean = tf.math.log(mean)
            tensor_log_prior = self.log_prior(x_scaled)
            prob = tf.reshape(log_mean, shape=tensor_log_prior.shape) + tensor_log_prior
            all_predictions.append(prob)
        return tf.math.exp(all_predictions)
    

    @tf.function(experimental_compile=True)
    def sample_density(self, num_results, num_burnin_steps, initialise_chains, sampler):
        """
        Run sampler over marginal posterior surface. This returns the posterior samples.
        :param num_results: final number of posterior samples
        :param num_burnin_steps: number of burn-ins
        :param initialise_chains: initial sampling point for all chains. This has size (Q, 1).
        :param sampler: Default is MCMC, alternative option is HMC
        """
        assert initialise_chains.shape[0] > 1, 'number of chains is 1, at least two chains must be used when sampling'
        p0 = tf.convert_to_tensor(initialise_chains, dtype=float_type)

        if sampler == 'MCMC':
            inner_sampler = tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=self.log_prob)

            sampler = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=inner_sampler,
                bijector=self.bijector_list)

        if sampler == 'HMC':
            inner_sampler = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.log_prob,
                step_size=0.00052,
                num_leapfrog_steps=5)

            transitioned_kernel = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=inner_sampler,
                bijector=self.bijector_list)

            sampler = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=transitioned_kernel,
                                                        num_adaptation_steps=int(0.8 * num_burnin_steps))

        if sampler == 'NUTS':
            sampler = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=self.log_prob,
                step_size=0.00052)
        print('Running sampler with {}'.format(str(sampler.__class__).split('.')[-1].split('\'')[0]))

        results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=1,  # thinning of chains
            current_state=p0,
            kernel=sampler
        )

        states = results.all_states
        return states


def scale_inverse_transform(whatever_scaler, X):
    """
    Wrapper around sklearn scaler (inverse) that doesn't involve numpy
    :param parameters_scaler: An sklearn minmax scaler
    :param X: Data we are transforming
    """
    x_new = tf.identity(X)
    try:
        x_new -= whatever_scaler.min_
        x_new /= whatever_scaler.scale_
    except AttributeError:
        x_new *= whatever_scaler.scale_
        x_new += whatever_scaler.mean_
    return x_new
