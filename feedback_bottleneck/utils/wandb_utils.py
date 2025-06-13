import html
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import wandb

from feedback_bottleneck.utils.utils import retry


def init_wandb(args):
    """
    Must call initialization of Wandb before summary writer is initialized, otherwise
    sync_tensorboard does not work.
    """

    if args.wandb_unique_id is None:
        # if we're going to restart the experiment, this will be saved to a json file
        args.wandb_unique_id = f'{args.exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'

    if not args.log_wandb:
        print("Weights and Biases integration disabled")
        return

    wandb_unique_id = args.wandb_unique_id

    print(
        "Weights and Biases integration enabled. Project: %s, user: %s, group: %s, unique_id: %s",
        args.wandb_project_name,
        args.wandb_entity,
        args.wandb_group,
        wandb_unique_id,
    )

    import wandb

    # this can fail occasionally, so we try a couple more times
    @retry(3, exceptions=(Exception,))
    def init_wandb_func():
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            id=wandb_unique_id,
            name=wandb_unique_id,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            tags=args.wandb_tags,
            resume="allow",
            settings=wandb.Settings(start_method="fork"),
            config=vars(args),
        )

    print("Initializing WandB...")
    try:
        init_wandb_func()
    except Exception as exc:
        print(f"Could not initialize WandB! {exc}")
        raise

    wandb.define_metric("train/samples")
    wandb.define_metric("*", step_metric="train/samples")


def finish_wandb(args):
    if args.log_wandb:
        import wandb

        wandb.run.finish()


def _parse_observation(text):
    """Refined regex logic from your notebook."""
    keys_list = [
        "map",
        "overview",
        "map description",
        "prayer status",
        "inventory",
        "message",
        "language observation",
        "cursor",
    ]
    pattern = r"(" + "|".join(keys_list) + r"):"
    parts = re.split(pattern, text)
    if parts:
        parts.pop(0)
    parts = [part.strip() for part in parts]
    data = dict(zip(parts[0::2], parts[1::2]))
    if "map" in data:
        data["map"] = data["map"].replace("><", "")
    return data


def _format_obs_to_html(text_short, text_long):
    """Converts raw observation text into a nice HTML block."""
    full_text = str(text_short) + "\n" + str(text_long)
    obs_data = _parse_observation(full_text)

    html_parts = []
    # Order of display
    keys = ["map", "map description", "overview", "message", "inventory", "prayer status", "language observation"]

    for k in keys:
        if k in obs_data and obs_data[k]:
            clean_val = html.escape(obs_data[k]).replace("\n", "<br>")
            # Use a monospace font for the map to preserve alignment
            style = "font-family: monospace; white-space: pre;" if k == "map" else ""
            html_parts.append(
                f"<div style='margin-bottom: 10px;'>"
                f"<strong>{k.upper()}</strong><br>"
                f"<div style='background: #f0f0f0; padding: 5px; {style}'>{clean_val}</div>"
                f"</div>"
            )
    return "".join(html_parts)


def create_episode_html(df_episode, episode_id):
    """Generates a standalone HTML string for a single episode with dual sliders."""

    # --- 1. Extract Columns ---
    raw_plans = df_episode["plans"].tolist()
    rewards = df_episode["rewards"].tolist()
    accum_rewards = np.cumsum(rewards).tolist()
    text_actions = df_episode["text_actions"].tolist()
    raw_outputs = df_episode["raw_outputs"].tolist()
    terms = df_episode["terms"].tolist()
    truncs = df_episode["truncs"].tolist()
    short_ctx = df_episode["short_term_context"].tolist()
    long_ctx = df_episode["long_term_context"].tolist()

    total_steps = len(raw_plans)

    # --- 2. Build Plan Segments (for Plan Slider & Chart) ---
    segments = []
    current_plan = "No Plan"
    seg_start = 0

    for i, plan in enumerate(raw_plans):
        plan = str(plan) if plan is not None else ""
        if plan and plan != current_plan:
            if i > 0:
                segments.append(
                    {
                        "plan_idx": len(segments),
                        "start": seg_start,
                        "end": i - 1,
                        "duration": i - seg_start,
                        "label": f"Plan {len(segments)}",
                    }
                )
            current_plan = plan
            seg_start = i

    # Final segment
    segments.append(
        {
            "plan_idx": len(segments),
            "start": seg_start,
            "end": total_steps - 1,
            "duration": total_steps - seg_start,
            "label": f"Plan {len(segments)}",
        }
    )

    # --- 3. Build Per-Step JSON Data (for Action Slider) ---
    step_data = []
    for idx in range(total_steps):
        # Format strings
        plan_clean = html.escape(str(raw_plans[idx]) if raw_plans[idx] else "").replace(".", ".<br>")
        raw_out_clean = html.escape(str(raw_outputs[idx]) if raw_outputs[idx] else "").replace("\n", "<br>")
        obs_html = _format_obs_to_html(short_ctx[idx], long_ctx[idx])

        step_data.append(
            {
                "idx": idx,
                "plan_html": plan_clean,
                "raw_output_html": raw_out_clean,
                "obs_html": obs_html,
                "action": str(text_actions[idx]),
                "stats": {
                    "term": bool(terms[idx]),
                    "trunc": bool(truncs[idx]),
                    "reward": round(rewards[idx], 4),
                    "reward_acc": round(accum_rewards[idx], 4),
                },
            }
        )

    # Prepare JS Payloads
    json_steps = json.dumps(step_data)
    json_segments = json.dumps(segments)
    max_duration = max(s["duration"] for s in segments) if segments else 0

    # --- 4. Construct HTML ---
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; padding: 20px; background-color: #fdfdfd; }}
            .container {{ max_width: 1100px; margin: 0 auto; }}

            /* Chart */
            .chart-box {{ display: flex; align-items: flex-end; height: 80px; gap: 1px; margin-bottom: 15px; border-bottom: 1px solid #ccc; }}
            .bar {{ background-color: #87CEFA; flex: 1; min-width: 2px; transition: background 0.2s; cursor: pointer; }}
            .bar:hover {{ background-color: #4682B4; }}
            .bar.active {{ background-color: #FF8C00; border: 1px solid #d35400; }}

            /* Controls Layout */
            .controls {{ background: #eee; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
            .control-row {{ display: flex; align-items: center; margin-bottom: 10px; }}
            .control-row label {{ width: 120px; font-weight: bold; }}
            .control-row input {{ flex: 1; }}
            .control-row span {{ width: 60px; text-align: right; display: inline-block; }}

            /* Panels Grid */
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .panel {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; background: white; }}
            .panel h4 {{ margin-top: 0; border-bottom: 2px solid #eee; padding-bottom: 5px; color: #444; }}

            .stat-box {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; font-size: 0.9em; }}
            .stat-item {{ background: #f9f9f9; padding: 5px 10px; border-radius: 4px; border: 1px solid #eee; }}

            .content-box {{ font-family: monospace; white-space: pre-wrap; font-size: 0.9em; max-height: 400px; overflow-y: auto; }}
            .plan-box {{ background: #fff8dc; border-left: 4px solid orange; padding: 10px; }}
            .raw-box {{ background: #eef; border-left: 4px solid slateblue; padding: 10px; color: #333; }}
            .action-highlight {{ font-size: 1.1em; color: #d35400; font-weight: bold; padding: 10px; background: #fff0e0; border-radius: 4px; text-align: center; margin-bottom: 10px;}}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Episode {episode_id}</h2>

            <div id="chart" class="chart-box"></div>

            <div class="controls">
                <div class="control-row">
                    <label>Plan Segment:</label>
                    <input type="range" id="plan-slider" min="0" max="{len(segments) - 1}" value="0">
                    <span id="plan-lbl">0</span>
                </div>
                <div class="control-row">
                    <label>Action Step:</label>
                    <input type="range" id="step-slider" min="0" max="{total_steps - 1}" value="0">
                    <span id="step-lbl">0</span>
                </div>
            </div>

            <div class="panel" style="margin-bottom: 20px;">
                <div class="action-highlight">
                    ACTION: <span id="d-action"></span>
                </div>
                <div class="stat-box">
                    <div class="stat-item">Frame: <b id="d-frame">0</b></div>
                    <div class="stat-item">Reward (Step): <b id="d-rew">0</b></div>
                    <div class="stat-item">Reward (Acc): <b id="d-rew-acc">0</b></div>
                    <div class="stat-item">Term: <b id="d-term"></b></div>
                    <div class="stat-item">Trunc: <b id="d-trunc"></b></div>
                </div>
            </div>

            <div class="grid">
                <div>
                    <div class="panel" style="margin-bottom: 20px;">
                        <h4>CURRENT PLAN</h4>
                        <div id="d-plan" class="content-box plan-box"></div>
                    </div>
                    <div class="panel">
                        <h4>OBSERVATION</h4>
                        <div id="d-obs" class="content-box"></div>
                    </div>
                </div>

                <div>
                    <div class="panel">
                        <h4>RAW LLM OUTPUT</h4>
                        <div id="d-raw" class="content-box raw-box"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const steps = {json_steps};
            const segments = {json_segments};
            const maxDuration = {max_duration};

            const planSlider = document.getElementById('plan-slider');
            const stepSlider = document.getElementById('step-slider');
            const chart = document.getElementById('chart');

            // DOM Elements
            const els = {{
                planLbl: document.getElementById('plan-lbl'),
                stepLbl: document.getElementById('step-lbl'),
                action: document.getElementById('d-action'),
                frame: document.getElementById('d-frame'),
                rew: document.getElementById('d-rew'),
                rewAcc: document.getElementById('d-rew-acc'),
                term: document.getElementById('d-term'),
                trunc: document.getElementById('d-trunc'),
                plan: document.getElementById('d-plan'),
                obs: document.getElementById('d-obs'),
                raw: document.getElementById('d-raw')
            }};

            // 1. Render Chart (Bars represent Plan Segments)
            segments.forEach((seg, i) => {{
                let bar = document.createElement('div');
                bar.className = 'bar';
                bar.title = 'Plan ' + i + ' (Steps: ' + seg.duration + ')';
                let h = (seg.duration / maxDuration) * 100;
                bar.style.height = Math.max(h, 10) + '%';
                bar.id = 'bar-' + i;

                // Clicking bar jumps to that plan
                bar.onclick = () => {{
                    planSlider.value = i;
                    syncSliders('plan');
                }};

                chart.appendChild(bar);
            }});

            // 2. State Management
            function updateView(stepIdx) {{
                const d = steps[stepIdx];

                // Update Action Slider UI
                stepSlider.value = stepIdx;
                els.stepLbl.innerText = stepIdx;

                // Stats
                els.frame.innerText = d.idx;
                els.action.innerText = d.action;
                els.rew.innerText = d.stats.reward;
                els.rewAcc.innerText = d.stats.reward_acc;
                els.term.innerText = d.stats.term;
                els.trunc.innerText = d.stats.trunc;

                // Content
                els.plan.innerHTML = d.plan_html;
                els.obs.innerHTML = d.obs_html;
                els.raw.innerHTML = d.raw_output_html;

                // Highlight correct Plan Bar
                // Find which segment this step belongs to
                const activeSeg = segments.find(s => stepIdx >= s.start && stepIdx <= s.end);
                if (activeSeg) {{
                    document.querySelectorAll('.bar').forEach(b => b.classList.remove('active'));
                    document.getElementById('bar-' + activeSeg.plan_idx).classList.add('active');

                    // Sync Plan Slider Label
                    els.planLbl.innerText = activeSeg.plan_idx;
                    planSlider.value = activeSeg.plan_idx;
                }}
            }}

            function syncSliders(source) {{
                if (source === 'plan') {{
                    // User moved Plan Slider -> Jump Action Slider to start of that plan
                    const segIdx = parseInt(planSlider.value);
                    const seg = segments[segIdx];
                    updateView(seg.start);
                }} else if (source === 'step') {{
                    // User moved Action Slider -> Just update view
                    updateView(parseInt(stepSlider.value));
                }}
            }}

            // 3. Listeners
            planSlider.addEventListener('input', () => syncSliders('plan'));
            stepSlider.addEventListener('input', () => syncSliders('step'));

            // Init
            updateView(0);
        </script>
    </body>
    </html>
    """
    return html_content


def log_wandb_html_trajectories(episode_df, num_episodes=5):
    """Logs the top N episodes as interactive HTML files to WandB."""

    unique_ids = episode_df["episode_id"].unique()

    # Process a few sample episodes (HTML logs can be large)
    for eid in unique_ids[:num_episodes]:
        # Filter dataframe for this episode
        subset = episode_df[episode_df["episode_id"] == eid].reset_index(drop=True)

        # Generate the HTML content
        html_str = create_episode_html(subset, eid)

        # Log to WandB
        wandb.log({f"trajectory/episode_{eid}": wandb.Html(html_str, inject=False)})
