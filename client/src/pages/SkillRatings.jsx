import { useState } from "react";
import { Link, useLocation } from "react-router";
import {
    FiArrowLeft,
    FiChevronRight,
} from "react-icons/fi";
import { useNavigate } from "react-router";

const sampleSkillsToRate = ["Docker", "Kubernetes", "System Design"];
const SLIDER_THUMB_SIZE = 20;

function extractSkillsToRate(evaluationResult) {
    const skillGaps = evaluationResult?.skill_gaps;

    if (Array.isArray(skillGaps) && skillGaps.length > 0) {
        return skillGaps.map((skill) =>
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
    const backendUrl = import.meta.env.VITE_BACKEND_URL;
    const navigate = useNavigate();
    function handleSkillRatingSubmit() {
        const response = fetch(`${backendUrl}/api/submit-ratings`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                ratings,
            }),
        });
        if (response.ok) {
            // Handle successful submission (e.g., navigate to learning path)
            const result = response.json();
            navigate("/learning-path", { state: { learningPath: result.learningPath } });
        } else {
            console.error("Error submitting ratings:", response.statusText);
            // Handle error (e.g., display error message to user)
        }
    }

    return (
        <main className="min-h-screen bg-[radial-gradient(circle_at_top,#ecfeff,#dbeafe_38%,#cbd5e1_72%,#0f172a_155%)] px-4 py-4 text-slate-900 sm:px-6 sm:py-6">
            <div className="mx-auto flex max-w-6xl flex-col gap-5">
                <header className="rounded-4xl border border-white/60 bg-white/80 p-5 shadow-xl shadow-slate-900/5 backdrop-blur sm:p-6">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                        <div className="max-w-3xl">
                            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-700">
                                Skill Synapse
                            </p>
                            <h1 className="mt-2 font-serif text-3xl font-semibold tracking-tight text-slate-950 sm:text-4xl">
                                Self-Rating Checkpoint
                            </h1>
                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                We can decide which skills need more context
                                from you. This screen gives the user a quick way to
                                add confidence ratings before the next stage to minimize mistakes in the learning path recommendations.
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

                <section className="rounded-4xl border border-slate-200/70 bg-white/85 p-5 shadow-xl shadow-slate-900/5 backdrop-blur">
                        <div className="flex items-center justify-between gap-3">
                            <div>
                                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                                    Rate your skills
                                </p>
                                <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
                                    Confidence snapshot
                                </h2>
                            </div>
                        </div>

                        <div className="mt-5 grid gap-4">
                            {skillsToRate.map((skill) => {
                                const selected = ratings[skill];
                                const sliderValue = selected ?? 0;

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
                                                min="0"
                                                max="10"
                                                step="1"
                                                value={sliderValue}
                                                onChange={(event) =>
                                                    setRatings((current) => ({
                                                        ...current,
                                                        [skill]: Number(event.target.value),
                                                    }))
                                                }
                                                className="h-2 w-full cursor-pointer appearance-none rounded-full bg-sky-100 accent-sky-600 [&::-moz-range-thumb]:h-5 [&::-moz-range-thumb]:w-5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-sky-600 [&::-webkit-slider-runnable-track]:h-2 [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-runnable-track]:bg-sky-100 [&::-webkit-slider-thumb]:-mt-1.5 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white [&::-webkit-slider-thumb]:bg-sky-600 [&::-webkit-slider-thumb]:shadow-md"
                                            />
                                            <div className="relative mt-2 h-4 text-xs font-medium text-slate-500">
                                                {Array.from({ length: 11 }, (_, index) => (
                                                    <span
                                                        key={index}
                                                        className="absolute top-0 -translate-x-1/2"
                                                        style={{
                                                            left: `calc(${SLIDER_THUMB_SIZE / 2}px + ((100% - ${SLIDER_THUMB_SIZE}px) * ${index} / 10))`,
                                                        }}
                                                    >
                                                        {index}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    </article>
                                );
                            })}
                        </div>

                        <div className="mt-6 flex justify-end">
                            <button
                                onClick={handleSkillRatingSubmit}
                                className="inline-flex items-center justify-center gap-2 rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-700"
                            >
                                Continue to learning path
                                <FiChevronRight className="text-base" />
                            </button>
                        </div>
                </section>
            </div>
        </main>
    );
}
