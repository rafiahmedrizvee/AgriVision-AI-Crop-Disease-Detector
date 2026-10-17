
const API_BASE_URL = 'http://127.0.0.1:8000'

export async function predictDisease(imageFile) {
  const formData = new FormData()
  formData.append('file', imageFile)

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  })

  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    throw new Error(
      data.detail || data.message || `Request failed: ${response.status}`
    )
  }

  return data
}
