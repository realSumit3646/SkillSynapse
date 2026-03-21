import { Link, useLocation } from "react-router";
import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    PolarAngleAxis,
    PolarGrid,
    PolarRadiusAxis,
    Radar,
    RadarChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import {
    FiArrowLeft,
    FiAward,
    FiChevronRight,
    FiClock,
    FiTarget,
    FiTrendingUp,
} from "react-icons/fi";

const sampleEvaluationResult = {
    all_skills: {
        Python: {
            score: 8.2,
            difficulty: 2,
            days: 14,
            unlock_power: 10,
        },
        Docker: {
            score: 5.2,
            difficulty: 6,
            days: 14,
            unlock_power: 3,
        },
        Kubernetes: {
            score: 3.5,
            difficulty: 8,
            days: 30,
            unlock_power: 2,
        },
        SQL: {
            score: 7.4,
            difficulty: 3,
            days: 10,
            unlock_power: 7,
        },
    },
};

const MAX_RADAR_SKILLS = 8;
const MAX_BAR_CHART_SKILLS = 10;

function parseDays(details) {
    const directDays = Number(details.days);
    if (Number.isFinite(directDays) && directDays > 0) {
        return directDays;
    }

    const time = String(details.time ?? "").trim().toLowerCase();
    const match = time.match(/(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months)/);

    if (!match) {
        return 0;
    }

    const value = Number(match[1]);
    const unit = match[2];

    if (unit.startsWith("week")) {
        return Math.round(value * 7);
    }

    if (unit.startsWith("month")) {
        return Math.round(value * 30);
    }

    return Math.round(value);
}

function formatDays(days) {
    return `${days} day${days === 1 ? "" : "s"}`;
}

function RadarTooltip({ active, payload }) {
    if (!active || !payload?.length) {
        return null;
    }

    const skill = payload[0].payload;

    return (
        <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-xl">
            <p className="text-sm font-semibold text-slate-900">{skill.skill}</p>
            <p className="mt-2 text-sm text-slate-600">Score: {skill.score}/10</p>
            <p className="text-sm text-slate-600">
                Difficulty: {skill.difficulty}/10
            </p>
            <p className="text-sm text-slate-600">Days: {skill.days}</p>
            <p className="text-sm text-slate-600">
                Unlock power: {skill.unlockPower}
            </p>
        </div>
    );
}

function ChartTooltip({ active, payload, label }) {
    if (!active || !payload?.length) {
        return null;
    }

    return (
        <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-xl">
            {label ? (
                <p className="text-sm font-semibold text-slate-900">{label}</p>
            ) : null}
            {payload.map((entry) => (
                <p
                    key={`${entry.name}-${entry.dataKey}`}
                    className="text-sm"
                    style={{ color: entry.color }}
                >
                    {entry.name}: {entry.value}
                </p>
            ))}
        </div>
    );
}

export default function Evaluate() {
    const { state } = useLocation();
    const evaluationResult =
        state?.evaluationResult && Object.keys(state.evaluationResult).length
            ? state.evaluationResult
            : sampleEvaluationResult;
    const skillGaps = Array.isArray(evaluationResult.skill_gaps)
        ? evaluationResult.skill_gaps
        : [];
    const hasSkillGaps = skillGaps.length > 0;

    const skills = Object.entries(evaluationResult.all_skills ?? {}).map(
        ([skill, details]) => {
            const days = parseDays(details);
            const unlockPower = Number(details.unlock_power ?? 0);

            return {
                skill,
                score: Number(details.score ?? 0),
                difficulty: Number(details.difficulty ?? 0),
                days,
                unlockPower,
                unlockPerDay: days > 0 ? Number((unlockPower / days).toFixed(2)) : 0,
            };
        },
    );

    const averageScore = skills.length
        ? (
              skills.reduce((total, skill) => total + skill.score, 0) /
              skills.length
          ).toFixed(1)
        : "0.0";
    const totalLearningDays = skills.reduce((total, skill) => total + skill.days, 0);
    const totalUnlockPower = skills.reduce(
        (total, skill) => total + skill.unlockPower,
        0,
    );

    const fastestWin = skills.reduce(
        (best, skill) =>
            skill.unlockPerDay > best.unlockPerDay ? skill : best,
        skills[0] ?? { skill: "None", unlockPerDay: 0 },
    );

    const radarSkills = [...skills]
        .sort((a, b) => b.score - a.score)
        .slice(0, MAX_RADAR_SKILLS);

    const timeComparisonData = [...skills]
        .map((skill) => ({
            skill: skill.skill,
            days: skill.days,
            score: skill.score,
            unlockPerDay: skill.unlockPerDay,
        }))
        .sort((a, b) => a.days - b.days)
        .slice(0, MAX_BAR_CHART_SKILLS);

    const unlockPotentialData = [...skills]
        .map((skill) => ({
            skill: skill.skill,
            unlockPower: skill.unlockPower,
            days: skill.days,
            score: skill.score,
        }))
        .sort((a, b) => b.unlockPower - a.unlockPower)
        .slice(0, MAX_BAR_CHART_SKILLS);
    const effortPayoffData = [...skills]
        .map((skill) => ({
            skill: skill.skill,
            difficulty: skill.difficulty,
            unlockPower: skill.unlockPower,
        }))
        .sort((a, b) => b.unlockPower - a.unlockPower)
        .slice(0, MAX_BAR_CHART_SKILLS);

    const maxUnlockPower = Math.max(
        ...skills.map((skill) => skill.unlockPower),
        1,
    );

    return (
        <main className="min-h-screen bg-[radial-gradient(circle_at_top,#eff6ff,#dbeafe_34%,#cbd5e1_72%,#0f172a_150%)] px-4 py-4 text-slate-900 sm:px-6 sm:py-6">
            <div className="mx-auto flex max-w-7xl flex-col gap-5">
                <header className="rounded-4xl border border-white/60 bg-white/80 p-5 shadow-xl shadow-slate-900/5 backdrop-blur sm:p-6">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                        <div className="max-w-3xl min-w-0">
                            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-700">
                                Skill Synapse
                            </p>
                            <h1 className="mt-2 font-serif text-3xl font-semibold tracking-tight text-slate-950 sm:text-4xl">
                                Skill Match Dashboard
                            </h1>
                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                Match quality, learning time, and upside are compared
                                in days so you can see which skills deserve your next move.
                            </p>
                        </div>

                        <Link
                            to="/"
                            className="inline-flex shrink-0 items-center gap-2 self-start rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-sky-300 hover:text-sky-700"
                        >
                            <FiArrowLeft className="text-base" />
                            Back to upload
                        </Link>
                    </div>
                </header>

                <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                    <article className="min-w-0 rounded-[1.75rem] border border-slate-200/70 bg-slate-950 p-4 text-white shadow-xl shadow-slate-900/10">
                        <div className="flex items-center gap-3">
                            <div className="rounded-2xl bg-sky-400/15 p-3 text-sky-300">
                                <FiTrendingUp className="text-lg" />
                            </div>
                            <div className="min-w-0">
                                <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                                    Average score
                                </p>
                                <p className="mt-1 text-2xl font-semibold">
                                    {averageScore}/10
                                </p>
                            </div>
                        </div>
                    </article>

                    <article className="min-w-0 rounded-[1.75rem] border border-slate-200/70 bg-slate-950 p-4 text-white shadow-xl shadow-slate-900/10">
                        <div className="flex items-center gap-3">
                            <div className="rounded-2xl bg-emerald-400/15 p-3 text-emerald-300">
                                <FiClock className="text-lg" />
                            </div>
                            <div className="min-w-0">
                                <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                                    Total learning days
                                </p>
                                <p className="mt-1 text-2xl font-semibold">
                                    {totalLearningDays}
                                </p>
                            </div>
                        </div>
                    </article>

                    <article className="min-w-0 rounded-[1.75rem] border border-slate-200/70 bg-slate-950 p-4 text-white shadow-xl shadow-slate-900/10">
                        <div className="flex items-center gap-3">
                            <div className="rounded-2xl bg-fuchsia-400/15 p-3 text-fuchsia-300">
                                <FiTarget className="text-lg" />
                            </div>
                            <div className="min-w-0">
                                <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                                    Total unlock power
                                </p>
                                <p className="mt-1 text-2xl font-semibold">
                                    {totalUnlockPower}
                                </p>
                            </div>
                        </div>
                    </article>

                    <article className="min-w-0 rounded-[1.75rem] border border-slate-200/70 bg-slate-950 p-4 text-white shadow-xl shadow-slate-900/10">
                        <div className="flex items-center gap-3">
                            <div className="rounded-2xl bg-amber-400/15 p-3 text-amber-300">
                                <FiAward className="text-lg" />
                            </div>
                            <div className="min-w-0">
                                <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                                    Best ROI skill
                                </p>
                                <p className="mt-1 truncate text-xl font-semibold">
                                    {fastestWin.skill}
                                </p>
                                <p className="text-xs text-slate-400">
                                    {fastestWin.unlockPerDay} unlock/day
                                </p>
                            </div>
                        </div>
                    </article>
                </section>

                <section className="grid gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(22rem,1.05fr)]">
                    <div className="grid gap-4">
                        <section className="flex min-w-0 flex-col overflow-hidden rounded-4xl border border-slate-200/70 bg-white/85 p-4 shadow-xl shadow-slate-900/5 backdrop-blur sm:p-5">
                            <div className="mb-3 flex items-center justify-between gap-4">
                                <div className="min-w-0">
                                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                                    Match profile
                                </p>
                                <h2 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
                                    Score radar by skill
                                </h2>
                                {skills.length > MAX_RADAR_SKILLS ? (
                                    <p className="mt-1 text-xs text-slate-500">
                                        Showing top {MAX_RADAR_SKILLS} skills by score for readability.
                                    </p>
                                ) : null}
                            </div>
                            <div className="shrink-0 rounded-full bg-sky-50 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-700">
                                {radarSkills.length}/{skills.length} skills
                            </div>
                        </div>

                        <div className="rounded-3xl bg-[linear-gradient(180deg,rgba(248,250,252,0.95),rgba(239,246,255,0.88))] p-3 sm:p-4">
                            <div className="mx-auto aspect-square w-full max-w-100 min-w-0">
                                <ResponsiveContainer width="100%" height="100%">
                                        <RadarChart data={radarSkills} outerRadius="69%">
                                            <PolarGrid gridType="polygon" stroke="#cbd5e1" />
                                            <PolarAngleAxis
                                                dataKey="skill"
                                                tick={{
                                                    fill: "#0f172a",
                                                    fontSize: 12,
                                                    fontWeight: 600,
                                                }}
                                            />
                                            <PolarRadiusAxis
                                                domain={[0, 10]}
                                                tickCount={6}
                                                tick={{
                                                    fill: "#64748b",
                                                    fontSize: 10,
                                                }}
                                                axisLine={false}
                                            />
                                            <Tooltip content={<RadarTooltip />} />
                                            <Radar
                                                name="Match score"
                                                dataKey="score"
                                                stroke="#0284c7"
                                                fill="#38bdf8"
                                                fillOpacity={0.45}
                                                strokeWidth={3}
                                            />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </section>

                        <article className="flex min-w-0 flex-col overflow-hidden rounded-4xl border border-slate-200/70 bg-white/85 p-4 shadow-xl shadow-slate-900/5 backdrop-blur">
                            <div>
                                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                                    Effort vs payoff
                                </p>
                                <h3 className="mt-1 text-lg font-semibold text-slate-950">
                                    Difficulty against unlock power
                                </h3>
                                <p className="mt-2 text-sm text-slate-500">
                                    This makes the tradeoff clearer: higher bars on unlock power
                                    are attractive, but difficulty shows the effort you will need
                                    to invest to get there.
                                </p>
                                {skills.length > MAX_BAR_CHART_SKILLS ? (
                                    <p className="mt-2 text-xs text-slate-500">
                                        Showing top {MAX_BAR_CHART_SKILLS} skills by unlock power.
                                    </p>
                                ) : null}
                            </div>

                            <div className="mt-4 h-80 min-w-0 overflow-hidden rounded-3xl bg-[linear-gradient(180deg,rgba(248,250,252,0.95),rgba(254,249,195,0.65))] p-3 sm:h-88">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={effortPayoffData} barGap={10}>
                                        <CartesianGrid
                                            strokeDasharray="3 3"
                                            stroke="#e5e7eb"
                                            vertical={false}
                                        />
                                        <XAxis
                                            dataKey="skill"
                                            tick={{ fontSize: 11, fill: "#475569" }}
                                            axisLine={false}
                                            tickLine={false}
                                        />
                                        <YAxis
                                            tick={{ fontSize: 11, fill: "#64748b" }}
                                            axisLine={false}
                                            tickLine={false}
                                        />
                                        <Tooltip content={<ChartTooltip />} />
                                        <Legend />
                                        <Bar
                                            dataKey="difficulty"
                                            name="Difficulty"
                                            fill="#f59e0b"
                                            radius={[10, 10, 0, 0]}
                                        />
                                        <Bar
                                            dataKey="unlockPower"
                                            name="Unlock power"
                                            fill="#10b981"
                                            radius={[10, 10, 0, 0]}
                                        />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </article>
                    </div>

                    <aside className="grid min-w-0 gap-4">
                        <section className="rounded-4xl border border-slate-200/70 bg-white/85 p-5 shadow-xl shadow-slate-900/5 backdrop-blur">
                            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                                {hasSkillGaps ? "Continue the flow" : "Ready for the next step"}
                            </p>
                            <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
                                {hasSkillGaps
                                    ? "Add your own confidence ratings"
                                    : "Good work, you are in a strong position"}
                            </h2>
                            <p className="mt-3 text-sm leading-6 text-slate-600">
                                {hasSkillGaps
                                    ? "The next step asks you to rate the skills that need more personal context."
                                    : "There are no flagged skill gaps in this evaluation result, so you can move directly into the learning path."}
                            </p>

                            <div className="mt-5 rounded-[1.75rem] border border-sky-200 bg-[linear-gradient(135deg,#0f172a,#0f766e)] p-5 text-white shadow-lg shadow-sky-200/40">
                                <p className="text-[11px] uppercase tracking-[0.22em] text-sky-200">
                                    {hasSkillGaps ? "Self-rating checkpoint" : "Direct path unlocked"}
                                </p>
                                <p className="mt-2 max-w-sm text-sm leading-6 text-slate-100">
                                    {hasSkillGaps
                                        ? 'Bring in your own judgment before we move to the next route. This is where the system asks, "How strong do you actually feel in these skills?"'
                                        : "Your evaluation looks healthy enough to skip the extra self-rating checkpoint and head straight into the next learning plan."}
                                </p>
                                <Link
                                    to={hasSkillGaps ? "/skill-rating" : "/learning-path"}
                                    state={{ evaluationResult }}
                                    className="mt-5 inline-flex items-center gap-2 rounded-full bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-sky-100"
                                >
                                    {hasSkillGaps
                                        ? "Rate flagged skills"
                                        : "Go to learning path"}
                                    <FiChevronRight className="text-base" />
                                </Link>
                            </div>
                        </section>

                        <section className="flex min-w-0 flex-col overflow-hidden rounded-4xl border border-slate-200/70 bg-white/85 p-4 shadow-xl shadow-slate-900/5 backdrop-blur">
                            <div className="mb-3 flex items-center justify-between gap-3">
                                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                                    Skill breakdown
                                </p>
                                <span className="shrink-0 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                    {skills.length} total skills
                                </span>
                            </div>

                            <div className="grid max-h-144 gap-3 overflow-y-auto pr-1">
                                {skills.map((skill) => (
                                    <article
                                        key={skill.skill}
                                        className="min-w-0 rounded-3xl border border-slate-200 bg-slate-50/90 p-3"
                                    >
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="min-w-0">
                                                <h3 className="truncate text-base font-semibold text-slate-900">
                                                    {skill.skill}
                                                </h3>
                                                <p className="truncate text-xs text-slate-500">
                                                    {formatDays(skill.days)}
                                                </p>
                                            </div>
                                            <div className="shrink-0 rounded-full bg-slate-900 px-3 py-1 text-xs font-semibold text-white">
                                                {skill.score}/10
                                            </div>
                                        </div>

                                        <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-slate-600">
                                            <div className="truncate rounded-2xl bg-white px-3 py-2">
                                                Difficulty {skill.difficulty}/10
                                            </div>
                                            <div className="truncate rounded-2xl bg-white px-3 py-2">
                                                Unlock {skill.unlockPower}
                                            </div>
                                            <div className="truncate rounded-2xl bg-white px-3 py-2">
                                                ROI {skill.unlockPerDay}/day
                                            </div>
                                        </div>
                                    </article>
                                ))}
                            </div>
                        </section>
                    </aside>
                </section>

                <section className="grid gap-4 lg:grid-cols-2">
                    <article className="flex min-w-0 flex-col overflow-hidden rounded-4xl border border-slate-200/70 bg-white/85 p-4 shadow-xl shadow-slate-900/5 backdrop-blur">
                        <div>
                            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                                Learning time
                            </p>
                            <h3 className="mt-1 text-lg font-semibold text-slate-950">
                                Fastest to slowest skills
                            </h3>
                            <p className="mt-2 text-sm text-slate-500">
                                A cleaner time view so you can immediately see which
                                skills are quicker wins and which ones demand a longer runway.
                            </p>
                            {skills.length > MAX_BAR_CHART_SKILLS ? (
                                <p className="mt-2 text-xs text-slate-500">
                                    Showing the {MAX_BAR_CHART_SKILLS} quickest skills.
                                </p>
                            ) : null}
                        </div>

                        <div className="mt-4 h-88 min-w-0 overflow-hidden rounded-3xl bg-[linear-gradient(180deg,rgba(248,250,252,0.95),rgba(224,242,254,0.75))] p-3 sm:h-96">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={timeComparisonData}
                                    layout="vertical"
                                    margin={{ top: 4, right: 12, left: 8, bottom: 4 }}
                                >
                                    <CartesianGrid
                                        strokeDasharray="3 3"
                                        stroke="#dbeafe"
                                        horizontal={false}
                                    />
                                    <XAxis
                                        type="number"
                                        tick={{ fontSize: 11, fill: "#64748b" }}
                                        axisLine={false}
                                        tickLine={false}
                                        name="Days"
                                    />
                                    <YAxis
                                        type="category"
                                        dataKey="skill"
                                        width={84}
                                        tick={{ fontSize: 11, fill: "#475569" }}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <Tooltip content={<ChartTooltip />} />
                                    <Legend />
                                    <Bar
                                        dataKey="days"
                                        name="Days"
                                        fill="#0ea5e9"
                                        radius={[0, 10, 10, 0]}
                                    />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </article>

                    <article className="flex min-w-0 flex-col overflow-hidden rounded-4xl border border-slate-200/70 bg-white/85 p-4 shadow-xl shadow-slate-900/5 backdrop-blur">
                        <div>
                            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                                Unlock potential
                            </p>
                            <h3 className="mt-1 text-lg font-semibold text-slate-950">
                                Highest unlock power
                            </h3>
                            <p className="mt-2 text-sm text-slate-500">
                                A ranked view of which skills could open the most doors,
                                without making you decode a scatter plot first.
                            </p>
                            {skills.length > MAX_BAR_CHART_SKILLS ? (
                                <p className="mt-2 text-xs text-slate-500">
                                    Showing top {MAX_BAR_CHART_SKILLS} skills by unlock power.
                                </p>
                            ) : null}
                        </div>

                        <div className="mt-4 h-88 min-w-0 overflow-hidden rounded-3xl bg-[linear-gradient(180deg,rgba(248,250,252,0.95),rgba(240,253,244,0.75))] p-3 sm:h-96">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={unlockPotentialData}
                                    layout="vertical"
                                    margin={{ top: 4, right: 12, left: 8, bottom: 4 }}
                                >
                                    <CartesianGrid
                                        strokeDasharray="3 3"
                                        stroke="#d1fae5"
                                        horizontal={false}
                                    />
                                    <XAxis
                                        type="number"
                                        domain={[0, maxUnlockPower + 2]}
                                        tick={{ fontSize: 11, fill: "#64748b" }}
                                        axisLine={false}
                                        tickLine={false}
                                        name="Unlock power"
                                    />
                                    <YAxis
                                        type="category"
                                        dataKey="skill"
                                        width={84}
                                        tick={{ fontSize: 11, fill: "#475569" }}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <Tooltip content={<ChartTooltip />} />
                                    <Legend />
                                    <Bar
                                        dataKey="unlockPower"
                                        name="Unlock power"
                                        fill="#10b981"
                                        radius={[0, 10, 10, 0]}
                                    />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </article>
                </section>
            </div>
        </main>
    );
}
