import type { Metadata } from "next";
import {
  ArrowDown,
  ArrowUpRight,
  Check,
  CircleDot,
  Code2,
} from "lucide-react";
import { Equation, InlineMath } from "../components/Math";
import modelSummary from "../public/data/model-summary.json";

export const metadata: Metadata = {
  title: "Storage Optionality Lab | Mathematical Specification",
  description:
    "A practical mathematical specification for gas storage valuation, LSMC control, reinforcement learning, hedging, and risk analysis.",
};

const sections = [
  ["scope", "1. Valuation problem"],
  ["lab", "2. Simulated model lab"],
  ["state", "3. State and information"],
  ["curve", "4. Forward-curve model"],
  ["physical", "5. Physical storage"],
  ["cashflow", "6. Cashflow and terminal"],
  ["dynamic", "7. Dynamic program"],
  ["lsmc", "8. LSMC specification"],
  ["rl", "9. RL formulation"],
  ["extrinsic", "10. Extrinsic value"],
  ["hedging", "11. Hedges and risk"],
  ["implementation", "12. Model boundary"],
  ["roadmap", "13. Valuation roadmap"],
  ["references", "References"],
] as const;

function SectionHeading({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <header className="section-heading">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p className="section-lede">{children}</p>
    </header>
  );
}

function Definition({
  term,
  children,
}: {
  term: string;
  children: React.ReactNode;
}) {
  return (
    <div className="definition-row">
      <dt>{term}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function LabFigure({
  src,
  alt,
  caption,
}: {
  src: string;
  alt: string;
  caption: React.ReactNode;
}) {
  return (
    <figure className="data-figure lab-figure">
      <img src={src} alt={alt} width={1600} height={900} />
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

export default function Home() {
  return (
    <main>
      <header className="topbar">
        <a className="wordmark" href="#top" aria-label="Storage Optionality Lab home">
          <span className="wordmark-mark">SO</span>
          <span>Storage Optionality Lab</span>
        </a>
        <nav className="toplinks" aria-label="Primary navigation">
          <a href="#roadmap">Valuation plan</a>
          <a
            className="github-link"
            href="https://github.com/wuxx1016"
            target="_blank"
            rel="noreferrer"
          >
            <Code2 size={17} aria-hidden="true" />
            GitHub
          </a>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="hero-inner">
          <p className="kicker">Research specification · Version 0.3</p>
          <h1>Gas storage as a curve-dependent real option.</h1>
          <p className="hero-copy">
            A mathematical foundation for valuing physical inventory flexibility,
            learning operating policies, and separating forward-curve hedges from
            basis and operational risk.
          </p>
          <Equation label="0">
            {String.raw`V_t(s)=\sup_{\pi\in\Pi}\;\mathbb{E}^{\mathbb{Q}}_t\!\left[\sum_{u=t}^{T-1}D_{t,u}\,C_u\!\left(s_u,\pi_u(s_u)\right)+D_{t,T}G(s_T)\right]`}
          </Equation>
          <div className="hero-meta">
            <span><CircleDot size={15} aria-hidden="true" /> Monthly toy model today</span>
            <span><Check size={15} aria-hidden="true" /> Simulated sensitivity plots added</span>
            <a href="#scope">Read the specification <ArrowDown size={15} aria-hidden="true" /></a>
          </div>
        </div>
      </section>

      <div className="article-grid">
        <aside className="toc" aria-label="Table of contents">
          <p className="toc-title">Contents</p>
          <nav>
            {sections.map(([id, title]) => (
              <a key={id} href={`#${id}`}>{title}</a>
            ))}
          </nav>
        </aside>

        <article className="paper">
          <section id="scope" className="paper-section">
            <SectionHeading eyebrow="01 · Valuation problem" title="Contract, measure, and objective">
              The storage is a finite-horizon stochastic control problem. Physical
              inventory is controlled through injection, withdrawal, or hold decisions;
              market risk enters through the forward curve and local basis.
            </SectionHeading>

            <p>
              Let decision dates be <InlineMath>{String.raw`t_0<\cdots<t_N=T`}</InlineMath>.
              At each date, the operator observes market and facility state
              <InlineMath>{String.raw` s_t`}</InlineMath>, chooses a feasible control
              <InlineMath>{String.raw` q_t`}</InlineMath>, receives cashflow
              <InlineMath>{String.raw` C_t`}</InlineMath>, and carries inventory into the
              next date. The valuation measure must be stated explicitly:
            </p>

            <dl className="definitions">
              <Definition term="Risk-neutral value">
                Use <InlineMath>{String.raw`\mathbb Q`}</InlineMath> with curves and
                tradable-factor dynamics calibrated to option and futures markets. This is
                the primary fair-value specification.
              </Definition>
              <Definition term="Real-world strategy">
                Use <InlineMath>{String.raw`\mathbb P`}</InlineMath> for expected operating
                P&amp;L, hedge backtests, liquidity costs, and risk limits. A statistical
                forecast is not automatically a fair-value model.
              </Definition>
            </dl>

            <aside className="judgment">
              <span>Modeling judgment</span>
              Report fair value and strategy value separately. Mixing historical drift with
              risk-neutral discounting can create apparent value that is really a directional
              forecast.
            </aside>
          </section>

          <section id="lab" className="paper-section lab-section">
            <SectionHeading eyebrow="02 · Simulated model lab" title="A runnable toy valuation workflow">
              The current code generates simulated state paths, fits an LSMC policy,
              evaluates it out of sample, and publishes diagnostics that explain which
              assumptions move value. These are research checks, not calibrated market
              marks.
            </SectionHeading>

            <div className="metric-strip" aria-label="Simulated model summary">
              <div>
                <span>Base LSMC value</span>
                <strong>{modelSummary.base_value}</strong>
                <small>standard error {modelSummary.base_stderr}</small>
              </div>
              <div>
                <span>Largest upside test</span>
                <strong>{modelSummary.best_scenario}</strong>
                <small>value {modelSummary.best_scenario_value}</small>
              </div>
              <div>
                <span>Largest downside test</span>
                <strong>{modelSummary.worst_scenario}</strong>
                <small>value {modelSummary.worst_scenario_value}</small>
              </div>
              <div>
                <span>Vol / mean-reversion range</span>
                <strong>{modelSummary.surface_min}-{modelSummary.surface_max}</strong>
                <small>surface min to max</small>
              </div>
            </div>

            <LabFigure
              src="/figures/lab/model-process.png"
              alt="Modeling process from market state and facility state to feasible actions, LSMC valuation, policy value, and risk views"
              caption="The website now follows the same modeling pipeline as the Python package: one simulated environment, common random numbers, fitted policy evaluation, and then risk diagnostics."
            />

            <div className="lab-grid two">
              <LabFigure
                src="/figures/lab/state-path-fan.png"
                alt="Fan charts for prompt price, prompt-next spread, winter-summer spread, and local basis"
                caption="The policy observes prompt price, short spread, seasonal spread, basis, volatility proxies, calendar state, and normalized inventory."
              />
              <LabFigure
                src="/figures/lab/value-sensitivity-bars.png"
                alt="Bar chart of LSMC value sensitivity to prompt volatility, spread volatility, mean reversion, basis risk, ratchets, and terminal target"
                caption="The first sensitivity pass shows that terminal constraints, mean reversion, basis risk, and ratchets can move value as much as volatility bumps."
              />
            </div>

            <LabFigure
              src="/figures/lab/spread-vol-mean-reversion-surface.png"
              alt="Heatmap of LSMC policy value over spread volatility and spread mean reversion"
              caption="This surface is the planned template for extrinsic-value studies: vary spread volatility and mean reversion jointly rather than reporting a single volatility shock."
            />

            <div className="lab-grid two">
              <LabFigure
                src="/figures/lab/initial-inventory-sensitivity.png"
                alt="Line chart showing policy value by initial inventory"
                caption="Initial inventory changes the reachable set of future states. The same forward curve can have different value depending on whether injection or withdrawal flexibility is scarce."
              />
              <LabFigure
                src="/figures/lab/hedge-delta-ladder.png"
                alt="Illustrative finite-difference monthly hedge delta ladder"
                caption="The hedge ladder is finite-difference based on the fitted value map. The production version should bump calibrated factors and map them to tradable monthly instruments."
              />
            </div>

            <LabFigure
              src="/figures/lab/policy-spread-maps.png"
              alt="Continuation value heatmaps over prompt-next spread and winter-summer spread at three inventory levels"
              caption="The continuation-value surface makes the inventory-spread interaction visible. This is a regression diagnostic: a policy chart that collapses to one action is a warning, not an insight."
            />

            <div className="insight-list">
              <div>
                <span>Finding 1</span>
                <p>Volatility is not a scalar value driver. Prompt volatility, seasonal spread volatility, and correlation with basis should be separated.</p>
              </div>
              <div>
                <span>Finding 2</span>
                <p>Ratchets and terminal rules control whether flexibility can actually be monetized. Higher volatility without reachable inventory states may not add much value.</p>
              </div>
              <div>
                <span>Finding 3</span>
                <p>Basis risk must be carried into hedge diagnostics. Hub deltas can look clean while local physical P&amp;L remains exposed.</p>
              </div>
            </div>

            <aside className="warning">
              <span>Current code boundary</span>
              The simulation uses fast path counts and smooth toy ratchets so it can run on a
              laptop. The final version should add calibrated market curves, daily facility
              rules, cross-fit LSMC, richer RL training, and hedge backtests before any
              chart is treated as a valuation result.
            </aside>
          </section>

          <section id="state" className="paper-section">
            <SectionHeading eyebrow="02 · State" title="Inventory meets the observable curve">
              The Markov state must be rich enough to explain future cashflows and feasible
              controls, but compact enough for stable conditional-value estimation.
            </SectionHeading>

            <Equation label="1" note="Recommended state used by both the LSMC and RL formulations.">
              {String.raw`s_t=\left(\bar I_t,\,F_t^{(0)},\,D_t^{pn},\,D_t^{ws},\,B_t,\,\sigma_t^p,\,\sigma_t^s,\,m_t,\,A_t\right)`}
            </Equation>

            <div className="symbol-grid">
              <div><code>Īₜ</code><span>Normalized inventory, <InlineMath>{String.raw`I_t/I_{\max}`}</InlineMath></span></div>
              <div><code>Fₜ⁽⁰⁾</code><span>Prompt hub forward</span></div>
              <div><code>Dₜᵖⁿ</code><span>Prompt minus next month</span></div>
              <div><code>Dₜʷˢ</code><span>Winter average minus summer average</span></div>
              <div><code>Bₜ</code><span>Local physical basis to the hedge hub</span></div>
              <div><code>σₜᵖ, σₜˢ</code><span>Prompt and spread volatility states</span></div>
              <div><code>mₜ</code><span>Month or season indicators</span></div>
              <div><code>Aₜ</code><span>Availability, outage, and constraint state</span></div>
            </div>

            <Equation label="2">
              {String.raw`D_t^{pn}=F(t,T_0)-F(t,T_1),\qquad D_t^{ws}=\frac{1}{|\mathcal W|}\sum_{j\in\mathcal W}F(t,T_j)-\frac{1}{|\mathcal S|}\sum_{j\in\mathcal S}F(t,T_j)`}
            </Equation>

            <figure className="data-figure">
              <img
                src="/figures/toy-forward-curve.png"
                alt="Illustrative seasonal gas forward curve by delivery month"
                width={1000}
                height={620}
              />
              <figcaption>
                Illustrative simulated curve. The shape supplies value signals; it is not a
                market calibration or a valuation result.
              </figcaption>
            </figure>
          </section>

          <section id="curve" className="paper-section">
            <SectionHeading eyebrow="03 · Market dynamics" title="A multifactor forward-curve model">
              A practical valuation model should move prompt, long-term level, seasonal
              spreads, volatility, and local basis independently but with calibrated
              correlation.
            </SectionHeading>

            <h3>Recommended risk-neutral specification</h3>
            <p>
              For delivery date <InlineMath>{String.raw`T_j`}</InlineMath>, use a
              multifactor lognormal forward model. Because the modeled instruments are
              forwards, the risk-neutral drift is zero before convexity and settlement
              adjustments.
            </p>

            <Equation label="3">
              {String.raw`\frac{dF(t,T_j)}{F(t,T_j)}=\sigma_L(T_j)\,dW_t^L+\sigma_P e^{-\kappa_P(T_j-t)}\,dW_t^P+\sigma_S g_S(T_j)\,dW_t^S`}
            </Equation>

            <p>
              The long factor moves the whole curve, the decaying prompt factor creates
              Samuelson-style front volatility and mean reversion, and the seasonal loading
              <InlineMath>{String.raw`g_S(T)`}</InlineMath> moves winter against summer.
              Under correlated shocks with instantaneous correlation matrix
              <InlineMath>{String.raw`R_t`}</InlineMath>, the exact one-step simulation is:
            </p>

            <Equation label="4">
              {String.raw`\log\frac{F(t+\Delta t,T_j)}{F(t,T_j)}=-\frac12\,\Sigma_j(t)\Delta t+\boldsymbol\sigma_j(t)^\top L_t\sqrt{\Delta t}\,\varepsilon_{t+1},\quad L_tL_t^\top=R_t`}
            </Equation>

            <Equation label="5">
              {String.raw`\Sigma_j(t)=\boldsymbol\sigma_j(t)^\top R_t\boldsymbol\sigma_j(t),\qquad \varepsilon_{t+1}\sim\mathcal N(0,I)`}
            </Equation>

            <h3>Stochastic volatility and spread variance</h3>
            <Equation label="6">
              {String.raw`dz_t=\kappa_v(\theta_v-z_t)dt+\nu_v\sqrt{z_t}\,dW_t^v,\qquad \boldsymbol\sigma_j(t)=\sqrt{z_t}\,\boldsymbol\sigma_j^0(t)`}
            </Equation>

            <p>
              Storage responds to relative prices, so the variance of a spread matters more
              directly than the variance of either leg in isolation:
            </p>

            <Equation label="7">
              {String.raw`\operatorname{Var}_t(\Delta F_i-\Delta F_j)=\sigma_i^2+\sigma_j^2-2\rho_{ij}\sigma_i\sigma_j`}
            </Equation>

            <aside className="judgment">
              <span>Implication</span>
              Raising prompt volatility does not guarantee a higher storage value. If both
              delivery legs move together, actionable spread volatility can remain small;
              stronger mean reversion and ratchet access can matter more.
            </aside>

            <h3>Local basis</h3>
            <Equation label="8">
              {String.raw`P_t^{loc}=F_t^{(0)}+B_t,\qquad dB_t=\kappa_B(\theta_B-B_t)dt+\sigma_B\,dW_t^B+J_t^B\,dN_t`}
            </Equation>
            <p>
              A mean-reverting basis factor captures ordinary location risk; an optional jump
              term represents congestion or operational dislocations. Basis may be correlated
              with hub prices, weather, and storage availability.
            </p>

            <figure className="data-figure">
              <img
                src="/figures/simulated-spreads.png"
                alt="Illustrative simulated prompt-next, winter-summer, and local basis paths"
                width={1000}
                height={620}
              />
              <figcaption>
                Illustrative state paths. Final calibration will target the covariance and
                mean-reversion of tradable monthly spreads and the unhedgeable local basis.
              </figcaption>
            </figure>
          </section>

          <section id="physical" className="paper-section">
            <SectionHeading eyebrow="04 · Facility" title="Inventory dynamics, losses, and ratchets">
              Physical flexibility is state dependent. The same price signal has different
              value when the cavern is empty, full, slow, unavailable, or close to a ratchet
              breakpoint.
            </SectionHeading>

            <Equation label="9">
              {String.raw`I_{t+1}=(1-\ell_t)I_t+\eta_{in}q_t^+-\frac{q_t^-}{\eta_{out}},\qquad 0\le I_{t+1}\le I_{\max}`}
            </Equation>

            <p>
              Here <InlineMath>{String.raw`q_t^+,q_t^-\ge0`}</InlineMath> are scheduled
              injection and withdrawal, <InlineMath>{String.raw`\eta_{in},\eta_{out}`}</InlineMath>
              are efficiencies, and <InlineMath>{String.raw`\ell_t`}</InlineMath> is inventory
              loss or fuel usage. Simultaneous injection and withdrawal is excluded.
            </p>

            <Equation label="10">
              {String.raw`0\le q_t^+\le A_t\,r_{in}(I_t),\qquad 0\le q_t^-\le A_t\,r_{out}(I_t),\qquad q_t^+q_t^-=0`}
            </Equation>

            <p>
              Production ratchets should be contract tables, interpolated piecewise linearly
              rather than globally smoothed. With knots
              <InlineMath>{String.raw`(I_k,r_k)`}</InlineMath>:
            </p>

            <Equation label="11">
              {String.raw`r(I)=r_k+\frac{r_{k+1}-r_k}{I_{k+1}-I_k}(I-I_k),\qquad I\in[I_k,I_{k+1}]`}
            </Equation>

            <div className="two-column-note">
              <div>
                <h4>Injection optionality</h4>
                <p>Largest at low inventory, but only if future withdrawal capacity remains sufficient to monetize the gas.</p>
              </div>
              <div>
                <h4>Withdrawal optionality</h4>
                <p>Largest at high inventory, but constrained by deliverability, minimum inventory, and terminal obligations.</p>
              </div>
            </div>
          </section>

          <section id="cashflow" className="paper-section">
            <SectionHeading eyebrow="05 · Economics" title="Cashflow and terminal conditions">
              Cashflows must settle against the physical location while hedges settle against
              tradable hubs. Terminal treatment should reflect the actual contract, not serve
              as a numerical convenience.
            </SectionHeading>

            <Equation label="12">
              {String.raw`C_t=-q_t^+\left(P_t^{loc}+c_{in,t}\right)+q_t^-\left(P_t^{loc}-c_{out,t}\right)-c_{fix,t}-c_{switch}\mathbf 1_{a_t\ne a_{t-1}}`}
            </Equation>

            <p>Three common terminal specifications cover most research cases:</p>
            <div className="formula-list">
              <div><span>Liquidation</span><InlineMath>{String.raw`G(I_T)=\alpha P_T^{loc}I_T`}</InlineMath></div>
              <div><span>Target penalty</span><InlineMath>{String.raw`G(I_T)=\alpha P_T^{loc}I_T-\lambda_T|I_T-I^*|`}</InlineMath></div>
              <div><span>Hard cycle</span><InlineMath>{String.raw`I_T=I^*`}</InlineMath></div>
            </div>

            <aside className="warning">
              <span>Validation check</span>
              Re-run the valuation under each economically plausible terminal rule. A large
              change is not numerical noise; it indicates material residual inventory option
              value or an underspecified commercial obligation.
            </aside>
          </section>

          <section id="dynamic" className="paper-section">
            <SectionHeading eyebrow="06 · Control" title="The Bellman recursion">
              Every method in this project solves the same constrained control problem. LSMC
              approximates continuation values; RL approximates a value or policy map.
            </SectionHeading>

            <Equation label="13">
              {String.raw`V_t(s_t)=\max_{a\in\mathcal A(s_t)}\left\{C_t(s_t,a)+D_{t,t+1}\,\mathbb E_t^{\mathbb Q}\!\left[V_{t+1}(s_{t+1})\mid s_t,a\right]\right\}`}
            </Equation>

            <Equation label="14">
              {String.raw`V_T(s_T)=G(I_T,P_T^{loc}),\qquad D_{t,u}=\exp\!\left(-\int_t^u r_v\,dv\right)`}
            </Equation>

            <p>
              A useful implementation works with the post-decision state
              <InlineMath>{String.raw`s_t^a`}</InlineMath>: inventory is updated by the chosen
              flow while the next market shock has not yet arrived. This makes the
              continuation target cleaner and exposes action-specific feasibility.
            </p>
          </section>

          <section id="lsmc" className="paper-section">
            <SectionHeading eyebrow="07 · Regression Monte Carlo" title="LSMC with curve and inventory interactions">
              The core regression should encode the economics of storage without relying on
              an uncontrolled polynomial expansion of every forward tenor.
            </SectionHeading>

            <Equation label="15">
              {String.raw`\widehat C_t(s_t^a)=\phi(s_t^a)^\top\widehat\beta_t,\qquad \widehat\beta_t=\arg\min_\beta\sum_{n\in\mathcal T}\left(Y_{t+1}^{(n)}-\phi(s_t^{a,(n)})^\top\beta\right)^2+\lambda\|\beta\|_2^2`}
            </Equation>

            <Equation label="16" note="Features are standardized within each time step; one month dummy is omitted.">
              {String.raw`\phi=\left[1,\bar I,\bar I^2,\log F^{(0)},D^{pn},D^{ws},B,\sigma^p,\sigma^s,\bar I D^{pn},\bar I D^{ws},\bar I B,\mathbf 1_{month},\mathbf 1_{season}\right]`}
            </Equation>

            <p>
              The inventory-spread interactions are structural, not decorative. A positive
              winter-summer spread can support injection when the cavern is empty, but can
              have little incremental value at full inventory because injection capacity has
              vanished.
            </p>

            <Equation label="17">
              {String.raw`\widehat Q_t(s,a)=C_t(s,a)+D_{t,t+1}\widehat C_t(s_t^a),\qquad \widehat\pi_t(s)=\arg\max_{a\in\mathcal A(s)}\widehat Q_t(s,a)`}
            </Equation>

            <h3>Bias controls</h3>
            <div className="method-steps">
              <div><span>01</span><p><strong>Cross-fit.</strong> Estimate continuation on one path fold and exercise on another to reduce in-sample look-ahead bias.</p></div>
              <div><span>02</span><p><strong>Use common paths.</strong> Compare policies and volatility scenarios with common random numbers.</p></div>
              <div><span>03</span><p><strong>Evaluate forward.</strong> Train backward, then value the frozen policy on fresh paths with confidence intervals.</p></div>
              <div><span>04</span><p><strong>Stress the basis.</strong> Inspect condition numbers, regularization, fold dispersion, and action-boundary stability.</p></div>
            </div>
          </section>

          <section id="rl" className="paper-section">
            <SectionHeading eyebrow="08 · Reinforcement learning" title="The same problem as a constrained MDP">
              RL becomes useful when the action space, constraint state, or objective is too
              nonlinear for a compact continuation regression. It still requires the same
              market model and an out-of-sample valuation protocol.
            </SectionHeading>

            <Equation label="18">
              {String.raw`\mathcal M=(\mathcal S,\mathcal A,P,r,\gamma),\quad r_t=D_{0,t}C_t,\quad r_T=D_{0,T}G(I_T,P_T^{loc})`}
            </Equation>

            <p>
              For a continuous actor output <InlineMath>{String.raw`u_t\in[-1,1]`}</InlineMath>,
              impose physical feasibility through a deterministic action map:
            </p>

            <Equation label="19">
              {String.raw`q_t(u_t,I_t)=\begin{cases}u_t\,r_{in}(I_t),&u_t\ge0,\\u_t\,r_{out}(I_t),&u_t<0,\end{cases}\quad\text{then clip to inventory bounds}`}
            </Equation>

            <Equation label="20">
              {String.raw`J(\theta)=\mathbb E\!\left[\sum_{t=0}^{T}\gamma^t r_t\right],\qquad \pi_\theta(s)=\arg\max_a Q_\theta(s,a)`}
            </Equation>

            <aside className="judgment">
              <span>Method choice</span>
              DQN is a transparent baseline for inject / hold / withdraw. PPO or SAC is more
              natural once partial rates, continuous nominations, and nonlinear fuel or
              switching costs are included. Algorithm complexity is not a substitute for a
              correctly calibrated environment.
            </aside>
          </section>

          <section id="extrinsic" className="paper-section">
            <SectionHeading eyebrow="09 · Optionality" title="Intrinsic and extrinsic value">
              Extrinsic value is the incremental value of adapting future actions to evolving
              prices, evaluated with identical physical constraints and terminal treatment.
            </SectionHeading>

            <Equation label="21">
              {String.raw`V^{intr}_0=\max_{\{q_t\}}\sum_{t=0}^{T-1}D_{0,t}C_t\!\left(q_t;F(0,T_t)\right)+D_{0,T}G(I_T)`}
            </Equation>

            <Equation label="22">
              {String.raw`V^{ext}_0=V^{stoch}_0-V^{intr}_0`}
            </Equation>

            <div className="driver-table" role="table" aria-label="Extrinsic value drivers">
              <div className="driver-head" role="row"><span>Driver</span><span>Primary channel</span><span>Expected diagnostic</span></div>
              <div role="row"><strong>Prompt volatility</strong><span>Short-horizon timing</span><span>Value vs front-factor vol bump</span></div>
              <div role="row"><strong>Spread volatility</strong><span>Seasonal cycling</span><span>Value vs winter-summer covariance</span></div>
              <div role="row"><strong>Mean reversion</strong><span>Repeatable dislocations</span><span>Two-way sensitivity; not assumed monotone</span></div>
              <div role="row"><strong>Inventory flexibility</strong><span>Reachable future states</span><span>Value by initial inventory and ratchet</span></div>
              <div role="row"><strong>Basis risk</strong><span>Physical / hedge mismatch</span><span>Unhedged P&amp;L and basis stress</span></div>
              <div role="row"><strong>Outages</strong><span>Lost exercise opportunities</span><span>Availability scenario loss</span></div>
            </div>
          </section>

          <section id="hedging" className="paper-section">
            <SectionHeading eyebrow="10 · Hedge and risk" title="From value to a tradable hedge ladder">
              Storage is hedged with monthly or seasonal forwards, while basis and physical
              availability remain only partially hedgeable.
            </SectionHeading>

            <h3>Forward deltas</h3>
            <Equation label="23">
              {String.raw`\Delta_{t,j}=\frac{\partial V_t}{\partial F(t,T_j)}\approx\frac{V_t(F+\varepsilon e_j)-V_t(F-\varepsilon e_j)}{2\varepsilon}`}
            </Equation>

            <p>
              Bump-and-revalue should preserve the calibrated factor structure and use common
              random numbers. A raw independent bump to one tenor can create a curve shape the
              model would never generate.
            </p>

            <h3>Minimum-variance hedge</h3>
            <Equation label="24">
              {String.raw`h_t^*=\arg\min_h\operatorname{Var}_t\!\left(\Delta V_{t,t+1}-h^\top\Delta F_{t,t+1}\right)=\Sigma_{FF,t}^{-1}\Sigma_{FV,t}`}
            </Equation>

            <Equation label="25">
              {String.raw`\Delta\Pi_{t,t+1}=\Delta V_{t,t+1}-h_t^\top\Delta F_{t,t+1}-TC(h_t-h_{t-1})`}
            </Equation>

            <p>
              The residual distribution should be decomposed into basis, constraint, model,
              execution, and policy error. A small delta is not evidence of low risk if local
              basis or availability dominates the residual.
            </p>

            <div className="planned-plots">
              <h3>Planned risk views</h3>
              <div><span>01</span><p>Monthly delta ladder with prompt, winter, and summer buckets</p></div>
              <div><span>02</span><p>Hedged versus unhedged P&amp;L distributions and expected shortfall</p></div>
              <div><span>03</span><p>Inventory fan chart, action frequencies, and ratchet utilization</p></div>
              <div><span>04</span><p>Value sensitivity surface across spread vol, mean reversion, and basis vol</p></div>
              <div><span>05</span><p>Policy maps by inventory, prompt-next spread, winter-summer spread, and month</p></div>
            </div>
          </section>

          <section id="implementation" className="paper-section">
            <SectionHeading eyebrow="11 · Model boundary" title="What exists now, and what changes for valuation">
              The runnable package is a research scaffold. The final valuation model will
              preserve its state and policy interfaces while replacing the main simplifying
              assumptions.
            </SectionHeading>

            <div className="comparison-table" role="table" aria-label="Toy and final model comparison">
              <div className="comparison-head" role="row"><span>Component</span><span>Current runnable model</span><span>Final valuation specification</span></div>
              <div role="row"><strong>Time grid</strong><span>24 monthly steps</span><span>Daily nomination calendar and no-flow days</span></div>
              <div role="row"><strong>Curve</strong><span>13 simulated tenors; prompt OU + spread factor</span><span>Arbitrage-consistent multifactor forward dynamics calibrated to market</span></div>
              <div role="row"><strong>Volatility</strong><span>Latent volatility proxy</span><span>Option-implied level, term structure, and factor correlation</span></div>
              <div role="row"><strong>Control</strong><span>Maximum inject / hold / maximum withdraw</span><span>Continuous or contract nomination increments</span></div>
              <div role="row"><strong>Ratchets</strong><span>Smooth inventory-dependent rates</span><span>Contract tables, outages, fuel, and losses</span></div>
              <div role="row"><strong>Basis</strong><span>Mean-reverting local additive factor</span><span>Location-specific history, jumps, and hedge mapping</span></div>
              <div role="row"><strong>Validation</strong><span>LSMC, NumPy DQN, heuristic comparison</span><span>Cross-fit LSMC, RL challenger, dual bounds, and hedge backtest</span></div>
            </div>
          </section>

          <section id="roadmap" className="paper-section roadmap-section">
            <SectionHeading eyebrow="12 · Next build" title="Plan for the final valuation and insight layer">
              Each phase has a numerical acceptance gate. The website will only promote a
              result to an insight after the policy, valuation, and hedge diagnostics agree.
            </SectionHeading>

            <ol className="roadmap">
              <li>
                <span>Phase 1</span>
                <div><h3>Calibrate the market and facility</h3><p>Load the initial curve, estimate factor covariance and mean reversion, specify local basis, and encode contract ratchets and costs.</p><small>Gate: simulated moments, spread distributions, and ratchet tables reproduce calibration targets.</small></div>
              </li>
              <li>
                <span>Phase 2</span>
                <div><h3>Establish the value benchmark</h3><p>Compute rolling intrinsic, cross-fit LSMC, and a frozen-policy out-of-sample value with standard errors and terminal-value sensitivity.</p><small>Gate: stable estimates across seeds, folds, basis families, and inventory grids.</small></div>
              </li>
              <li>
                <span>Phase 3</span>
                <div><h3>Train the RL challenger</h3><p>Use the same paths, rewards, action constraints, and terminal rule. Compare policy surfaces and lower-bound values rather than training rewards.</p><small>Gate: RL outperforms simple policies and remains economically coherent under stress.</small></div>
              </li>
              <li>
                <span>Phase 4</span>
                <div><h3>Build hedges and risk attribution</h3><p>Produce forward deltas, minimum-variance hedge ladders, transaction-cost P&amp;L, basis residuals, and tail-risk scenarios.</p><small>Gate: hedge results hold on unseen paths and remain robust to factor and basis misspecification.</small></div>
              </li>
              <li>
                <span>Phase 5</span>
                <div><h3>Publish evidence-backed findings</h3><p>Add interactive value decomposition, sensitivity surfaces, policy maps, inventory fans, and hedged P&amp;L distributions to this site.</p><small>Gate: every chart states its measure, scenario, units, confidence interval, and model version.</small></div>
              </li>
            </ol>
          </section>

          <section id="references" className="paper-section references-section">
            <SectionHeading eyebrow="Sources" title="Primary references">
              The specification follows the classical LSMC storage literature and connects it
              to modern statistical learning, deep optimization, and hedging methods.
            </SectionHeading>

            <ol className="references">
              <li><span>01</span><p>Longstaff, F. A. &amp; Schwartz, E. S. (2001). <a href="https://doi.org/10.1093/rfs/14.1.113" target="_blank" rel="noreferrer">Valuing American Options by Simulation: A Simple Least-Squares Approach <ArrowUpRight size={13} /></a>. <em>Review of Financial Studies</em>, 14(1), 113-147.</p></li>
              <li><span>02</span><p>Boogert, A. &amp; de Jong, C. (2008). <a href="https://repub.eur.nl/pub/13895" target="_blank" rel="noreferrer">Gas Storage Valuation Using a Monte Carlo Method <ArrowUpRight size={13} /></a>. <em>Journal of Derivatives</em>, 15(3), 81-98.</p></li>
              <li><span>03</span><p>Boogert, A. &amp; de Jong, C. (2011). <a href="https://doi.org/10.21314/JEM.2011.067" target="_blank" rel="noreferrer">Gas Storage Valuation Using a Multifactor Price Process <ArrowUpRight size={13} /></a>. <em>Journal of Energy Markets</em>, 4(4).</p></li>
              <li><span>04</span><p>Warin, X. (2012). <a href="https://doi.org/10.1007/978-3-642-25746-9_14" target="_blank" rel="noreferrer">Gas Storage Hedging <ArrowUpRight size={13} /></a>. In <em>Numerical Methods in Finance</em>, 421-445.</p></li>
              <li><span>05</span><p>Ludkovski, M. &amp; Maheshwari, A. (2020). <a href="https://doi.org/10.1007/s12667-018-0318-4" target="_blank" rel="noreferrer">Simulation Methods for Stochastic Storage Problems: A Statistical Learning Perspective <ArrowUpRight size={13} /></a>. <em>Energy Systems</em>, 11, 377-415.</p></li>
              <li><span>06</span><p>Curin, N. et al. (2021). <a href="https://doi.org/10.1007/s10203-021-00363-6" target="_blank" rel="noreferrer">A Deep Learning Model for Gas Storage Optimization <ArrowUpRight size={13} /></a>. <em>Decisions in Economics and Finance</em>, 44, 1021-1037.</p></li>
              <li><span>07</span><p>Buehler, H., Gonon, L., Teichmann, J. &amp; Wood, B. (2019). <a href="https://doi.org/10.1080/14697688.2019.1571683" target="_blank" rel="noreferrer">Deep Hedging <ArrowUpRight size={13} /></a>. <em>Quantitative Finance</em>, 19(8), 1271-1291.</p></li>
            </ol>
          </section>
        </article>

        <aside className="status-rail" aria-label="Model status">
          <div className="status-block">
            <p className="status-title">Specification status</p>
            <p><span className="status-dot current" /> Math model drafted</p>
            <p><span className="status-dot" /> Market calibration</p>
            <p><span className="status-dot" /> Final valuation</p>
            <p><span className="status-dot" /> Hedge analytics</p>
          </div>
          <div className="status-block status-note">
            <p className="status-title">Current boundary</p>
            <p>Equations are production-oriented. Figures are simulated illustrations. No displayed number is a market valuation.</p>
          </div>
        </aside>
      </div>

      <footer>
        <div><span className="wordmark-mark">SO</span><p>Storage Optionality Lab<br /><small>Mathematical specification · 2026</small></p></div>
        <a href="https://github.com/wuxx1016" target="_blank" rel="noreferrer">github.com/wuxx1016 <ArrowUpRight size={14} /></a>
      </footer>
    </main>
  );
}
