import { useState } from 'react'

export default function App() {
  const [resumeFile, setResumeFile] = useState(null)
  const [jobDescription, setJobDescription] = useState('')

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] ?? null
    setResumeFile(file)
  }

  const handleSubmit = (event) => {
    event.preventDefault()
  }

  return (
    <main className="min-h-screen bg-slate-100 px-4 py-4 text-slate-900 sm:px-6 sm:py-6">
      <div className="mx-auto flex min-h-[calc(100vh-2rem)] max-w-4xl items-center justify-center sm:min-h-[calc(100vh-3rem)]">
        <section className="w-full rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
          <div className="mb-6 flex items-start justify-between gap-4">
            <div className="space-y-2">
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-700">
                Skill Synapse
              </p>
              <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
                Resume Match
              </h1> 
              <p className="max-w-xl text-sm leading-6 text-slate-600 sm:text-base">
                Upload a resume, paste the job description, and submit both in
                one step.
              </p>
            </div>
          </div>

          <form className="grid gap-5" onSubmit={handleSubmit}>
            <label
              htmlFor="resume-upload"
              className="block cursor-pointer rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-5 py-5 transition hover:border-sky-400 hover:bg-sky-50"
            >
              <div className="flex items-center gap-4">
                <div className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-slate-900 text-sm font-bold uppercase tracking-[0.2em] text-white">
                  CV
                </div>
                <div>
                  <p className="text-base font-semibold text-slate-900">
                    {resumeFile ? resumeFile.name : 'Upload resume'}
                  </p>
                  <p className="text-sm text-slate-500">
                    PDF or DOCX
                  </p>
                </div>
              </div>
              <input
                id="resume-upload"
                type="file"
                accept=".pdf,.doc,.docx"
                className="sr-only"
                onChange={handleFileChange}
              />
            </label>

            <div className="space-y-2">
              <label
                htmlFor="job-description"
                className="text-sm font-semibold text-slate-700"
              >
                Job Description
              </label>
              <textarea
                id="job-description"
                rows="7"
                value={jobDescription}
                onChange={(event) => setJobDescription(event.target.value)}
                placeholder="Paste the role description here..."
                className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm leading-6 text-slate-700 outline-none transition placeholder:text-slate-400 focus:border-sky-400 focus:ring-4 focus:ring-sky-100"
              />
            </div>

            <button
              type="submit"
              className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-600 focus:outline-none focus:ring-4 focus:ring-sky-200"
            >
              Submit
            </button>
          </form>
        </section>
      </div>
    </main>
  )
}
