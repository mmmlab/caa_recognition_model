## Notes on Diffusion-to-Bound Word Recognition Model

### Task Synopsis (double-check these details)

We are attempting to model a word recognition task in which individual observers were presented with a list of words to remember. 
Some ancillary details of the task:

- The words to be remembered were presented one at a time on a computer display.
- The words were presented in various colors and in various locations on the display.
- After completing the primary recognition task described below, observers who recognized a word were asked to recall these details of its presentation as well.

Critical details of the task:

- Observers were presented with a list consisting of intermingled target (old) and lure (new) words.
- The words were presented one at a time and, for each word, observers had to decide as quickly as possible:
	1. Whether the word was a target (old) or a lure (new) word.
	2. Their confidence in their decision in (1) regarding the word. This was indicated as number ranging from 1-3, with larger numbers indicating greater confidence.
	3. Whether they made the decision in (1) based primarily on a feeling of familiarity ("know") or whether they distinctly remembered seeing the word ("remember").

### Data and Model
The primary data that we are modeling consists of the outcome of each trial (individual word judgments), specified in terms of four items:

1. The classification judgment (old vs. new and remember vs. know) along with its detection-theoretic category (i.e., hit, false-alarm, correct-rejection, or miss).
2. The confidence rating of the judgment.
3. The response time (RT) required to make the classification judgment.
4. The identity of the word.


The primary goal of our model(s) is to simultaneously explain the pattern of classifications and the distribution of reaction times within each classification and confidence category using a diffusion to bound model of word recognition. A secondary goal is to examine the variability of recognition strength across individual target words.

Our modeling approach can be seen as an extension of the diffusion model approach introduced by Ratcliff (1978) as a tool for simultaneously explaining categorical judgments and speeded response times in discrimination experiments.

The basic assumptions of the diffusion process are that evidence for a discrimination judgment accumulates continuously and that this accumulation of evidence can be described by a Wiener diffusion process. This Wiener diffusion process is characterized by a normally distributed drift/accumulation rate. The mean drift rate determines the average slope of the information accumulation process, that is, the speed and direction of information accumulation. The assumption of a variable, normally distributed drift rate implies that repeated processing of the same stimulus across different trials will yield variable response times, and may sometimes lead to different (i.e., erroneous) classification responses. 
    
There are two models that we've been using. The best-fitting one is a single-process model (SPM) that treats the remember/know distinction as a random decision. An alternative model is a dual-process model (DPM) inspired by Wixted & Mickes (2010) that assumes that two separate processes, familiarity and recall, are responsible for the old/new decision and for the confidence rating, while only recall is responsible for the remember/know judgment. 

#### Data Preprocessing

In this study, we restrict our analyses to the aggregated performance across human observers. This requires standardizing RT distributions across individual observers. We accomplished this using the Vincentization procedure suggested by Ratcliff (1979). First the responses were divided into categories based on the four signal detection theoretic response classifications (i.e., "hits", "false alarms", "misses", and "correct rejections"). Then, for each of these categories we:
1. Computed the RT quantiles for individual observers.
2. Averaged these individual quantiles to compute Vincentized "group quantiles".
3. Used the empirical distribution functions to determine the cumulative probability associated with each RT for each individual observer.
4. Plugged the cumulative probabilities (3) into the corresponding group quantile functions (2) to compute standardized RTs.

The analyses that follow were computed using these standardized RTs.

#### The Single-Process Model
The single-process model assumes that a single response variable $X$ representing the accumulation of evidence (i.e., a "familiarity" signal) over time is responsible for both the old/new decision and the confidence decision. In particular, the accumulation of information is modeled as a Weiner diffusion process that starts at $z_{0}$ and results in a decision when the response variable reaches one of two collapsing bounds $b_{old}$ or $b_{new}$.
<div>
<img src="http://rci.rutgers.edu/~mmm431/misc/d2b_schematic_mod.png" style="background-color:white" height=500>
<br>
Figure 1: Single process model schematic
</div>

The model's free parameters include:

|Parameter | ML Estimate | Description|
| -------- | ------- | :--------- |
|$\mu_{old}$| 0.328 units/sec |the drift rate of the Weiner process (i.e., the average accumulation rate of the decision variable) for old (target) words|
|$\mu_{new}$| -0.272 units/sec | the drift rate of the Weiner process for new (lure) words|
|$D$| 0.413 units<sup>2</sup>/sec| the diffusion rate of the Weiner process| 
|$\tau$| 21.231 sec| the time constant of the exponentially collapsing bounds|
|$x_0$|-0.124 units| the starting location of the Weiner process (positive values effectively represent a bias toward "old" judgments, whereas negative values represent a bias toward "new" judgments)|
|$t_0$|0.569 sec| the time point (with respect to the start of the trial) at which accumulation of the familiarity signal begins|
|$\lambda$|0.503 sec | the average elapsed time between the old/new decision and the confidence rating judgment.|
|$c_1$| 0.970 units |the location of the boundary between medium (1) and high (2) confidence "old" judgments. The low confidence  boundary $c_0$ is assumed to occur at $x=0$ and is not a free parameter.|


The rate of accumulation of evidence or drift rate is assumed to be normally distributed for both targets (old words) and lures (new words):

- $X\sim N(\mu_{old}\delta t,\sqrt{2D\delta t})$ for targets
- $X\sim N(\mu_{new}\delta t,\sqrt{2D\delta t})$ for lures.

Where $D$ represents the diffusion constant for the Weiner process and $\delta t$ represents the duration of the elapsed time interval.

For simplicity, we assume that the "old" and "new" decision boundaries are symmetric [CITE]. We also assume that these boundaries collapse exponentially as a function of time, so that 
- $b(t) = \exp{\left[-\frac{t-t_0}{\tau}\right]}$ for any time $t > t_0$,
- $b(t_0) = 1$, and
- $b_{new}(t) = - b_{old}(t) = -b(t)$.

 Evidence for such a collapsing bound has been reported by several studies (Bowman, Kording and Gottfried, 2012; Cisek, Puskas, and El-Murr, 2009; Ditterich, 2006; Thura, Beauregard–Racine, Fradet, and Cisek, 2012) and has been interpreted both in terms of the reduced impact of incremental information in the face of substantial accumulated evidence (Bowman et al., 2012; Ditterich, 2006; Hanks et al., 2011) and in terms of the effect of an increase in temporal urgency signal (Cisek et al., 2009; Thura et al., 2012). In the current model, the collapsing bound explains the inverse relationship between response time and confidence observed in the empirical data.

 Solving the general case of the collapsing-bound model for response time distributions is not analytically tractable. In the current project, we used discrete time increments (with $\delta t$ = 10 ms) to approximate the response time distributions as follows.

We assume that at the start of evidence accumulation, at $t_0$ seconds, the position of the evidence variable, denoted $X_{t_0}$ is equal to $x_0$.  After an additional interval of $\delta t$ seconds have elapsed, the new location of the evidence variable $X_{t_0+\delta t}$ is defined by the probability distribution

$$p_{t_0+\delta t}\left(X_{t_0+\delta t}\right) = \phi\left[\frac{X_{t_0+\delta t}-(x_0+\mu\delta t)}{\sqrt{2D\delta t}}\right], $$

where $\phi[\cdot]$ represents the standard normal distribution. 

At each further time point $t+\delta t$, the proportion of responses for which the accumulated evidence reaches some location $X_{t+\delta t}$ can be computed by multiplying the probability of each location $-b(t)<X_t<b(t)$ that did not exceed one of the response bounds at the previous time step, by the conditional probability of its current position given that it was at location $X_t$ in the previous time step and marginalizing over this previous location. I.e.,

$$p_{t+\delta t}\left(X_{t+\delta t}\right) = \int_{-b(t)}^{b(t)} p_t(X_t) p(X_{t+\delta t}|X_t)dX_{t}. $$

Because the conditional distribution of the random drift is independent of starting location, this is equivalent to convolving the location distribution at the previous time point $t$ with the probability of drift over the interval $\delta t$.

$$ \begin{aligned}
p_{t+\delta t}\left(X_{t+\delta t}\right) &= \int_{-b(t)}^{b(t)}p_t(X_t)~\phi\left[\frac{X_{t+\delta t}-(X_t+\mu\delta t)}{\sqrt{2D\delta t}}\right]dX_{t}
\\
\\
&= p_t(X_t) \ast \phi\left[\frac{\Delta X-\mu\delta t}{\sqrt{2D\delta t}}\right],
~ \text{for } -b(t)<X_t<b(t), 
\end{aligned}
$$

where $\Delta X = X_{t+\delta t}-X_t$ represents the magnitude of the drift over the interval $\delta t$.

The probability of an "old" response occuring within a temporal interval is just the probability of exceeding the corresponding bound. I.e.,

$$p_{t+\delta t}(\text{old}) =  p_{t+\delta t}\left(X_{t+\delta t}>b(t)\right) = \int_{b(t)}^{\infty} p_{t+\delta t}\left(X_{t+\delta t}\right) dX_t.$$

Similarly, the probability of a "new" response is

$$p_{t+\delta t}(\text{new}) =  p_{t+\delta t}\left(X_{t+\delta t}<-b(t)\right) = \int^{-b(t)}_{-\infty} p_{t+\delta t}\left(X_{t+\delta t}\right) dX_t.$$

Confidence categorizations were made on the basis of the location interval in which the evidence variable appeared at the time of the confidence judgment. The dashed gray lines in Figure 1 illustrate the confidence intervals and bounds. If confidence judgments were made simultaneously with the old/new decisions, then the judged confidence would be a deterministic function of the collapsing decision bound, and thus of response time. However, the confidence judgment was in fact a separate judgment made after the old/new decision. Though the actual elapsed time between these two judgments varied, the model, for simplicity, assumes a fixed temporal offset of $\lambda$. The model also assumes that evidence continues to accumulate over the interval $\lambda$. The effect of this additional interval of accumulation is that the evidence variable continues to drift, so that its location at the time of the confidence judgment will, in general, differ from its location at the time of the old/new judgment, and will follow a normal distribution 

$$X_{T+\lambda} \sim N(X_T,\sqrt{2D\lambda}),$$

where $T$ is the response time for the old/new judgment, $X_T$ is the location of the evidence variable at the time of the old/new judgment, and $X_{T+\lambda}$ is the location of the evidence variable at the time of the confidence judgment.

For example, an evidence variable classified as "old" will, at the time of its classification $T$, have a location equal to the value of the old bound at that time. I.e.,

$$X_T = b(T).$$

However, when the confidence judgment is made (after a further $\lambda$ seconds have elapsed), its new position will be a random variable drawn from a normal distribution with mean $b(T)$ and standard deviation $\sqrt{2D\lambda}$.

We set the low (0) confidence boundary to the 0 evidence location. This is tantamount to assuming that a confidence of 0 indicates old/new judgments that were reversed by the time of the confidence judgment (e.g., an "old" decision for which the evidence was negative at the time of the confidence judgment). The medium/high confidence boundary was fit to the observed data.

Finally, in the single-process model, we did not distinguish between the processes leading to "remember" versus "know" judgments, since the response time distributions did not vary substantially between these two categories. We assessed the goodness of fit of a particular model by computing response time deciles for each combination of word type (target or lure), decision category (old or new), and confidence level (low, medium, or high). This defined the model in terms of a multinomial distribution with a total of 80 possible categories. We used this multinomial distribution to compute the log-likelihood of the observed human responses. The MLE model was defined by the parameters that minimized the negative log-likelihood, which was about 546.07 for the reported parameters.

Figure 2 shows the results of the results of fitting the overall new/old judgments.
<div>
<img src="http://rci.rutgers.edu/~mmm431/misc/old_new_res.png" style="background-color:white" height=400>
<br>
Figure 2: Single process model fit (old/new)
</div>

Figure 3 shows the result of fitting the the single process model, then modeling the remember/know judgment as random choice, with category probabilities equal to those measured in the data (i.e., with $p(\text{know}|\text{old}) = 0.5255$ and $p(\text{know}|\text{old}) = 0.4745$). In this case, the resulting multinomial distribution included 140 possible categories (as a result of the additional remember/know distinction) and the overall negative log-likelihood for the resulting model was approximately 778.74. Because the RT distributions for remember and know judgments are very similar overall, this augmented model appears to fit the data nearly as well as the simpler old/new model, with one exception: the proportion of observed high confidence "new" target words that are classified as "remember" judgments (red dashed curve) is much greater than predicted by the marginal remember/know probabilities (solid red curve).
<div>
<img src="http://rci.rutgers.edu/~mmm431/misc/rem_know_res.png" style="background-color:white" height=400>
<br>
Figure 3: Single process model fit (remember/know)
</div>