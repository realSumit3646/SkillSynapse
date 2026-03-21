import { useState } from "react";
import { Link, useLocation } from "react-router";
import {
    FiArrowLeft,
    FiChevronRight,
    FiSliders,
} from "react-icons/fi";

const sampleSkillsToRate = ["Docker", "Kubernetes", "System Design"];

function extractSkillsToRate(evaluationResult) {
    const backendSkills = evaluationResult?.skills_to_rate;

    if (Array.isArray(backendSkills) && backendSkills.length > 0) {
        return backendSkills.map((skill) =>
            typeof skill === "string" ? skill : skill.name ?? skill.skill ?? "Skill",
        );
    }

    return sampleSkillsToRate;
}

export default function SkillRatings() {
    const { state } = useLocation();
    const evaluationResult = state?.evaluationResult;
    const skillsToRate = extractSkillsToRate(evaluationResult);
    const [ratings, setRatings] = useState({});

    const completedCount = skillsToRate.filter((skill) => ratings[skill] !== undefined).length;

    return (
        <main className="min-h-screen bg-[radial-gradient(circle_at_top,_#ecfeff,_#dbeafe_38%,_#cbd5e1_72%,_#0f172a_155%)] px-4 py-4 text-slate-900 sm:px-6 sm:py-6">
            <div className="mx-auto flex max-w-6xl flex-col gap-5">
                <header className="rounded-[2rem] border border-white/60 bg-white/80 p-5 shadow-xl shadow-slate-900/5 backdrop-blur sm:p-6">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                        <div className="max-w-3xl">
                            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-700">
                                Skill Synapse
                            </p>
                            <h1 className="mt-2 font-serif text-3xl font-semibold tracking-tight text-slate-950 sm:text-4xl">
                                Self-Rating Checkpoint
                            </h1>
                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                The backend can decide which skills need more context
                                from you. This screen gives the user a quick way to
                                add confidence ratings before the next stage.
                            </p>
                        </div>

                        <Link
                            to="/evaluate"
                            state={{ evaluationResult }}
                            className="inline-flex items-center gap-2 self-start rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-sky-300 hover:text-sky-700"
                        >
                            <FiArrowLeft className="text-base" />
                            Back to dashboard
                        </Link>
                    </div>
                </header>

                <section className="grid gap-4 xl:grid-cols-[0.85fr_1.15fr]">
                    <aside className="grid gap-4">
                        <section className="rounded-[2rem] border border-slate-200/70 bg-slate-950 p-5 text-white shadow-xl shadow-slate-900/10">
                            <div className="flex items-start gap-3">
                                <div className="rounded-2xl bg-sky-400/15 p-3 text-sky-300">
                                    <FiSliders className="text-lg" />
                                </div>
                                <div>
                                    <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-300">
                                        Progress
                                    </p>
                                    <h2 className="mt-2 text-2xl font-semibold">
                                        {completedCount}/{skillsToRate.length} rated
                                    </h2>
                                    <p className="mt-2 text-sm leading-6 text-slate-300">
                                        Quick confidence signals are enough here.
                                        The goal is to add your perspective, not to
                                        test you again.
                                    </p>
                                </div>
                            </div>
                        </section>

                        <section className="rounded-[2rem] border border-slate-200/70 bg-white/85 p-5 shadow-xl shadow-slate-900/5 backdrop-blur">
                            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                                What this step does
                            </p>
                            <div className="mt-4 grid gap-3 text-sm leading-6 text-slate-600">
                                <div className="rounded-3xl bg-slate-50 p-4">
                                    Gives the learning flow some real-world context from you.
                                </div>
                                <div className="rounded-3xl bg-slate-50 p-4">
                                    Helps separate low visibility from low ability.
                                </div>
                                <div className="rounded-3xl bg-slate-50 p-4">
                                    Leaves room for backend-selected skills to plug in later.
                                </div>
                            </div>
                        </section>
                    </aside>

                    <section className="rounded-[2rem] border border-slate-200/70 bg-white/85 p-5 shadow-xl shadow-slate-900/5 backdrop-blur">
                        <div className="flex items-center justify-between gap-3">
                            <div>
                                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                                    Rate your skills
                                </p>
                                <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
                                    Confidence snapshot
                                </h2>
                            </div>
                            <div className="rounded-full bg-sky-50 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-sky-700">
                                1 to 10 scale
                            </div>
                        </div>

                        <div className="mt-5 grid gap-4">
                            {skillsToRate.map((skill) => {
                                const selected = ratings[skill];
                                const sliderValue = selected ?? 5;

                                return (
                                    <article
                                        key={skill}
                                        className="rounded-[1.75rem] border border-slate-200 bg-slate-50/85 p-4"
                                    >
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="min-w-0">
                                                <h3 className="text-lg font-semibold text-slate-900">
                                                    {skill}
                                                </h3>
                                                <p className="mt-1 text-sm text-slate-500">
                                                    Rate your current confidence level.
                                                </p>
                                            </div>
                                            <div className="rounded-full bg-slate-950 px-3 py-1 text-sm font-semibold text-white">
                                                {sliderValue}/10
                                            </div>
                                        </div>

                                        <div className="mt-5">
                                            <input
                                                type="range"
                                                min="1"
                                                max="10"
                                                step="1"
                                                value={sliderValue}
                                                onChange={(event) =>
                                                    setRatings((current) => ({
                                                        ...current,
                                                        [skill]: Number(event.target.value),
                                                    }))
                                                }
                                                className="h-2 w-full cursor-pointer appearance-none rounded-full bg-sky-100 accent-sky-600"
                                            />
                                            <div className="mt-2 grid grid-cols-10 text-center text-xs font-medium text-slate-500">
                                                {Array.from({ length: 10 }, (_, index) => (
                                                    <span key={index + 1}>{index + 1}</span>
                                                ))}
                                            </div>
                                        </div>
                                    </article>
                                );
                            })}
                        </div>

                        <div className="mt-6 flex flex-col gap-3 rounded-[1.75rem] border border-sky-200 bg-[linear-gradient(135deg,_#f8fafc,_#e0f2fe)] p-4 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                                <p className="text-sm font-semibold text-slate-900">
                                    Ratings are captured locally for now
                                </p>
                                <p className="mt-1 text-sm text-slate-600">
                                    This screen is ready for backend-provided skills whenever that payload is added.
                                </p>
                            </div>
                            <Link
                                to="/"
                                className="inline-flex items-center justify-center gap-2 rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-700"
                            >
                                Finish for now
                                <FiChevronRight className="text-base" />
                            </Link>
                        </div>
                    </section>
                </section>
            </div>
        </main>
    );
}
