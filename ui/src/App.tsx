import { useState, useCallback, useRef, useEffect } from 'react'
import './App.css'

interface Match {
  character: string
  score: number
}

interface PredictResponse {
  model_loaded: boolean
  results: Match[]
}

interface CharactersResponse {
  model_loaded: boolean
  characters: string[]
  total?: number
}

interface CharImagesResponse {
  character: string
  images: string[]
}

export default function App() {
  const [image, setImage] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [results, setResults] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [characters, setCharacters] = useState<CharactersResponse | null>(null)
  const [charFilter, setCharFilter] = useState('')
  const [selectedChar, setSelectedChar] = useState<string | null>(null)
  const [charImages, setCharImages] = useState<string[]>([])
  const [charImgLoading, setCharImgLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetch('/api/characters')
      .then((r) => r.json())
      .then((d: CharactersResponse) => setCharacters(d))
      .catch(() => null)
  }, [])

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith('image/')) return
    setFile(f)
    setImage(URL.createObjectURL(f))
    setResults(null)
    setError(null)
  }, [])

  const handleUrlAsFile = useCallback(async (url: string) => {
    try {
      const res = await fetch(url)
      const blob = await res.blob()
      const f = new File([blob], 'image.jpg', { type: blob.type || 'image/jpeg' })
      handleFile(f)
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch {
      // ignore
    }
  }, [handleFile])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    // 파일 드롭
    const f = e.dataTransfer.files[0]
    if (f) { handleFile(f); return }
    // 갤러리 이미지 드롭
    const url = e.dataTransfer.getData('text/plain')
    if (url) handleUrlAsFile(url)
  }, [handleFile, handleUrlAsFile])

  const onInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) handleFile(f)
  }

  const predict = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch('/api/predict', { method: 'POST', body: form })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`HTTP ${res.status}: ${text}`)
      }
      setResults(await res.json() as PredictResponse)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setImage(null)
    setFile(null)
    setResults(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const selectChar = async (c: string) => {
    if (selectedChar === c) { setSelectedChar(null); setCharImages([]); return }
    setSelectedChar(c)
    setCharImages([])
    setCharImgLoading(true)
    try {
      const res = await fetch(`/api/characters/${encodeURIComponent(c)}/images`)
      const d: CharImagesResponse = await res.json()
      setCharImages(d.images)
    } catch {
      setCharImages([])
    } finally {
      setCharImgLoading(false)
    }
  }

  const filteredChars = characters?.characters.filter((c) =>
    c.toLowerCase().includes(charFilter.toLowerCase())
  ) ?? []

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">애니메이션 캐릭터 검색</h1>
        <p className="subtitle">캐릭터 이미지 업로드 → 유사 캐릭터 찾기</p>
      </header>

      <main className="main">
        <div
          className={`dropzone${dragging ? ' dragging' : ''}${image ? ' has-image' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => !image && inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && !image && inputRef.current?.click()}
          aria-label="이미지 업로드"
        >
          {image ? (
            <img src={image} alt="업로드된 이미지" className="preview" />
          ) : (
            <div className="drop-hint">
              <div className="drop-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <polyline points="21 15 16 10 5 21" />
                </svg>
              </div>
              <span className="drop-text">이미지 드래그 또는 클릭하여 선택</span>
              <span className="drop-sub">갤러리 이미지를 여기로 드래그할 수도 있음</span>
            </div>
          )}
          <input ref={inputRef} type="file" accept="image/*" onChange={onInput} style={{ display: 'none' }} />
        </div>

        <div className="actions">
          {image && (
            <>
              <button className="btn btn-secondary" onClick={reset}>초기화</button>
              <button className="btn btn-primary" onClick={predict} disabled={loading}>
                {loading ? <span className="loading-text"><span className="spinner" />분석 중...</span> : '캐릭터 검색'}
              </button>
            </>
          )}
        </div>

        {error && <div className="alert alert-error"><strong>오류:</strong> {error}</div>}

        {results && !results.model_loaded && (
          <div className="alert alert-warn">
            모델 미로드 — <code>scripts/export.py</code> 실행 후 <code>checkpoints/model.onnx</code>, <code>index.faiss</code>, <code>labels.npy</code> 필요
          </div>
        )}

        {results?.model_loaded && results.results.length === 0 && (
          <div className="alert alert-warn">결과 없음 — 인덱스가 비어있거나 유사한 캐릭터 없음</div>
        )}

        {results?.model_loaded && results.results.length > 0 && (
          <section className="results">
            <h2 className="results-title">분류 결과</h2>
            <div className="result-list">
              {results.results.map((r, i) => (
                <div className="result-item" key={i}>
                  <span className="result-rank">#{i + 1}</span>
                  <div className="result-body">
                    <span className="result-name">{r.character}</span>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${(r.score * 100).toFixed(1)}%` }} />
                    </div>
                  </div>
                  <span className="result-score">{(r.score * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {characters && (
          <section className="char-section">
            <div className="char-header">
              <h2 className="results-title">
                분류 가능 캐릭터
                {characters.model_loaded && (
                  <span className="char-count">{characters.total ?? characters.characters.length}개</span>
                )}
              </h2>
              {characters.model_loaded && characters.characters.length > 0 && (
                <input
                  className="char-search"
                  type="text"
                  placeholder="캐릭터 검색..."
                  value={charFilter}
                  onChange={(e) => setCharFilter(e.target.value)}
                />
              )}
            </div>

            {!characters.model_loaded ? (
              <p className="char-empty">모델 미로드 — 인덱스 빌드 후 표시됩니다</p>
            ) : characters.characters.length === 0 ? (
              <p className="char-empty">인덱스가 비어있습니다</p>
            ) : (
              <div className="char-grid">
                {filteredChars.map((c) => (
                  <button
                    className={`char-chip${selectedChar === c ? ' selected' : ''}`}
                    key={c}
                    onClick={() => selectChar(c)}
                  >
                    {c}
                  </button>
                ))}
              </div>
            )}

            {selectedChar && (
              <div className="char-gallery">
                <div className="gallery-header">
                  <span className="gallery-title">{selectedChar}</span>
                  <span className="gallery-hint">이미지를 위 분석창으로 드래그하면 검색</span>
                </div>
                {charImgLoading ? (
                  <div className="gallery-loading"><span className="spinner" /></div>
                ) : charImages.length === 0 ? (
                  <p className="char-empty">이미지 없음</p>
                ) : (
                  <div className="gallery-grid">
                    {charImages.map((src) => (
                      <img
                        key={src}
                        src={src}
                        alt={selectedChar}
                        className="gallery-img"
                        draggable
                        onDragStart={(e) => {
                          e.dataTransfer.setData('text/plain', src)
                          e.dataTransfer.effectAllowed = 'copy'
                        }}
                        onClick={() => handleUrlAsFile(src)}
                        title="클릭 또는 드래그하여 검색"
                      />
                    ))}
                  </div>
                )}
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  )
}
