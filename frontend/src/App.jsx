
import { useState, useRef, useEffect } from 'react'
import { predictDisease } from './services/api'
import { diseaseGuides } from './data/diseaseGuides'
import {
  Leaf,
  Sprout,
  Upload,
  ScanSearch,
  ShieldCheck,
  CloudSun,
  ArrowRight,
  Menu,
  X,
  Image as ImageIcon,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
} from 'lucide-react'
import './App.css'

function App() {
  const [menuOpen, setMenuOpen] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef(null)

  const diseaseGuide = result?.predicted_class
    ? diseaseGuides[result.predicted_class]
    : null

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  function handleFileChange(event) {
    const file = event.target.files?.[0]
    if (!file) return

    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setError('Please select a JPG, PNG, or WEBP image.')
      event.target.value = ''
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('Image size must be less than 10 MB.')
      event.target.value = ''
      return
    }

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError('')
  }

  async function handlePredict() {
    if (!selectedFile || loading) {
      if (!selectedFile) setError('Please select a tomato leaf image first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const data = await predictDisease(selectedFile)

      if (data.success === false) {
        throw new Error(data.message || 'Prediction failed.')
      }

      if (!data.predicted_class) {
        throw new Error('The server did not return a prediction.')
      }

      setResult(data)
    } catch (err) {
      setError(
        err.message?.toLowerCase().includes('fetch')
          ? 'Cannot connect to the AI server. Check that FastAPI is running at http://127.0.0.1:8000 and CORS is configured.'
          : err.message || 'Prediction failed. Please try again.'
      )
    } finally {
      setLoading(false)
    }
  }

  function resetImage() {
    setSelectedFile(null)
    setPreviewUrl('')
    setResult(null)
    setError('')

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="app">
      <header className="navbar">
        <a className="brand" href="#home">
          <span className="brand-icon"><Leaf size={25} /></span>
          <span>AgriVision<span className="brand-ai"> AI</span></span>
        </a>

        <button
          className="menu-toggle"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle navigation"
          aria-expanded={menuOpen}
        >
          {menuOpen ? <X /> : <Menu />}
        </button>

        <nav className={menuOpen ? 'nav-links open' : 'nav-links'}>
          <a href="#home" onClick={() => setMenuOpen(false)}>Home</a>
          <a href="#detect" onClick={() => setMenuOpen(false)}>Detect Disease</a>
          <a href="#features" onClick={() => setMenuOpen(false)}>Features</a>
          <a href="#about" onClick={() => setMenuOpen(false)}>About</a>
        </nav>

        <a className="nav-cta" href="#detect">
          Get Started <ArrowRight size={16} />
        </a>
      </header>

      <main>
        <section className="hero" id="home">
          <div className="hero-content">
            <div className="eyebrow">
              <Sprout size={16} /> SMART FARMING, HEALTHIER CROPS
            </div>

            <h1>
              Protect Your Crops with <span>AI-Powered</span> Intelligence.
            </h1>

            <p className="hero-description">
              Identify tomato plant diseases from leaf images and get
              practical guidance to help you make smarter farming decisions.
            </p>

            <div className="hero-actions">
              <a className="primary-button" href="#detect">
                <ScanSearch size={19} /> Detect Plant Disease
              </a>
              <a className="secondary-button" href="#features">
                Explore Features <ArrowRight size={17} />
              </a>
            </div>

            <div className="trust-row">
              <ShieldCheck size={19} />
              <span>AI-assisted analysis</span>
              <span className="trust-dot">•</span>
              <span>Fast results</span>
            </div>
          </div>

          <div className="hero-visual">
            <div className="visual-glow" />
            <div className="plant-illustration">
              <Leaf size={100} strokeWidth={1.1} />
              <Sprout size={150} strokeWidth={0.9} />
              <Leaf size={78} strokeWidth={1.1} />
            </div>

            <div className="floating-card card-top">
              <span className="status-dot" />
              <div>
                <strong>Plant Analysis</strong>
                <p>Ready to scan</p>
              </div>
            </div>

            <div className="floating-card card-bottom">
              <span className="mini-icon"><ShieldCheck size={22} /></span>
              <div>
                <strong>Early Detection</strong>
                <p>Make informed decisions</p>
              </div>
            </div>
          </div>
        </section>

        <section className="stats-strip" aria-label="Project information">
          <div><strong>4</strong><span>Tomato Classes</span></div>
          <div><strong>AI</strong><span>Image Classification</span></div>
          <div><strong>Fast</strong><span>Prediction Results</span></div>
        </section>

        <section className="detect-section" id="detect">
          <div className="section-heading">
            <span className="eyebrow">
              <ScanSearch size={16} /> DISEASE DETECTION
            </span>
            <h2>Give Your Plant a <span>Health Check</span></h2>
            <p>
              Upload a clear tomato leaf photo to get an AI-generated
              prediction from your trained model.
            </p>
          </div>

          <div className="upload-placeholder">
            {!previewUrl ? (
              <>
                <div className="upload-icon"><ImageIcon size={32} /></div>
                <h3>Upload Your Tomato Leaf</h3>
                <p>Select a clear image of a tomato leaf for analysis.</p>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleFileChange}
                  hidden
                />

                <button
                  className="primary-button"
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={18} /> Choose Image
                </button>

                <small>JPG, PNG or WEBP · Maximum size 10 MB</small>
              </>
            ) : (
              <div className="image-preview">
                <img src={previewUrl} alt="Selected tomato leaf preview" />
                <p className="file-name">{selectedFile?.name}</p>

                <div className="preview-actions">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={loading}
                  >
                    <RefreshCw size={17} /> Change Image
                  </button>

                  <button
                    className="primary-button"
                    type="button"
                    onClick={handlePredict}
                    disabled={loading}
                  >
                    <ScanSearch size={18} />
                    {loading ? 'Analyzing...' : 'Predict Disease'}
                  </button>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleFileChange}
                  hidden
                />

                {loading && (
                  <div className="loading-state" role="status">
                    <span className="loading-spinner" />
                    <p>AI is analyzing your tomato leaf...</p>
                  </div>
                )}
              </div>
            )}

            {error && (
              <div className="error-message" role="alert">
                <AlertCircle size={19} />
                <span>{error}</span>
              </div>
            )}

            {result && (
              <div className="prediction-result" role="status">
                <CheckCircle2 size={32} />
                <h3>Analysis Complete</h3>
                <p className="predicted-class">
                  {result.predicted_class.replaceAll('_', ' ')}
                </p>
                <p className="confidence">
                  Confidence: {Number(result.confidence_percent).toFixed(2)}%
                </p>

                {result.top_predictions?.length > 0 && (
                  <div className="top-predictions">
                    <h4>Prediction Breakdown</h4>
                    {result.top_predictions.map((item) => (
                      <div className="prediction-row" key={item.class_name}>
                        <span>{item.class_name.replaceAll('_', ' ')}</span>
                        <strong>
                          {Number(item.confidence_percent).toFixed(2)}%
                        </strong>
                      </div>
                    ))}
                  </div>
                )}

                {diseaseGuide && (
                  <section className="guidance-card" aria-labelledby="guidance-title">
                    <div className="guidance-heading">
                      <div>
                        <span className="guidance-eyebrow">FARMER GUIDANCE / কৃষি পরামর্শ</span>
                        <h4 id="guidance-title">{diseaseGuide.title}</h4>
                        <p className="guidance-bangla-title">{diseaseGuide.titleBn}</p>
                      </div>
                      <span className={`guidance-status ${result.predicted_class === 'Tomato_healthy' ? 'is-healthy' : 'is-disease'}`}>
                        {diseaseGuide.status}
                      </span>
                    </div>

                    <p className="guidance-description">{diseaseGuide.description}</p>

                    <div className="guidance-columns">
                      <div className="guidance-list">
                        <h5>Possible symptoms / সম্ভাব্য লক্ষণ</h5>
                        <ul>
                          {diseaseGuide.symptoms.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="guidance-list">
                        <h5>What to do / করণীয়</h5>
                        <ul>
                          {diseaseGuide.actions.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="guidance-list">
                        <h5>Prevention / প্রতিরোধ</h5>
                        <ul>
                          {diseaseGuide.prevention.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                    <p className="guidance-note">
                      Important: This guidance is general information, not a confirmed diagnosis. For severe or spreading symptoms, consult your local agricultural officer.
                    </p>
                  </section>
                )}

                <p className="result-disclaimer">
                  This is an AI-generated estimate, not a confirmed diagnosis.
                  Check real field conditions and consult an agricultural
                  expert before deciding on treatment.
                </p>

                <button
                  className="secondary-button"
                  type="button"
                  onClick={resetImage}
                >
                  <RefreshCw size={17} /> Analyze Another Image
                </button>
              </div>
            )}
          </div>
        </section>

        <section className="features-section" id="features">
          <div className="section-heading">
            <span className="eyebrow">
              <Sprout size={16} /> BUILT FOR SMARTER FARMING
            </span>
            <h2>Tools for <span>Healthier Crops</span></h2>
            <p>A simple interface designed to make AI-assisted plant analysis accessible.</p>
          </div>

          <div className="feature-grid">
            <article className="feature-card">
              <div className="feature-icon"><ScanSearch size={25} /></div>
              <h3>AI Leaf Analysis</h3>
              <p>Analyze a tomato leaf image using your trained image classification model.</p>
            </article>

            <article className="feature-card">
              <div className="feature-icon"><ShieldCheck size={25} /></div>
              <h3>Disease Awareness</h3>
              <p>See the predicted class and model confidence to support further investigation.</p>
            </article>

            <article className="feature-card">
              <div className="feature-icon"><CloudSun size={25} /></div>
              <h3>Practical Guidance</h3>
              <p>Learn general crop-care practices and when to seek local agricultural advice.</p>
            </article>
          </div>
        </section>

        <section className="about-section" id="about">
          <div className="about-icon"><Sprout size={34} /></div>
          <div>
            <span className="eyebrow">ABOUT AGRIVISION AI</span>
            <h2>Technology supporting better farming.</h2>
            <p>
              AgriVision AI explores how computer vision can help identify
              common tomato leaf conditions from photographs. Predictions
              should be checked against real field conditions and expert
              agricultural advice.
            </p>
          </div>
        </section>
      </main>

      <footer className="footer">
        <a className="brand" href="#home">
          <span className="brand-icon"><Leaf size={22} /></span>
          <span>AgriVision<span className="brand-ai"> AI</span></span>
        </a>
        <p>Smarter insights for healthier crops.</p>
        <span>© {new Date().getFullYear()} AgriVision AI</span>
      </footer>
    </div>
  )
}

export default App
