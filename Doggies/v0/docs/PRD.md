# Doggies — Product Requirements Document

## 1. Overview

**Product:** Doggies — AI-Powered Dog Adoption Assistant
**Location:** Koh Phangan, Thailand
**Mission:** Get more stray dogs adopted by making the process effortless, emotional, and available 24/7 in any language.

### The Problem

Koh Phangan has a significant stray dog population. A local shelter run by a single person struggles with:

1. **Visibility** — People don't know which dogs need homes
2. **Matching** — No scalable way to connect the right dog with the right adopter
3. **Funding** — Difficult to keep donors engaged over time
4. **Operations** — One admin doing everything manually

### The Solution

A Telegram bot powered by an AI agent that:
- Engages potential adopters in warm, natural conversation in any language
- Learns about their lifestyle and preferences
- Recommends matched dogs with emotionally compelling, personalized pitches
- Books shelter visits via Google Calendar
- Routes donors to one-time or recurring donation links
- Remembers every interaction to nurture relationships over time

---

## 2. Target Users

| User | Description | Primary Goal |
|------|------------|--------------|
| **Potential Adopter** | Expats, digital nomads, Thai locals on Koh Phangan | Find a dog that fits their life |
| **Potential Donor** | Tourists, remote followers, animal lovers | Support the shelter financially |
| **Shelter Admin** | Single operator of the shelter | Get notified of interest, manage visits |

### User Personas

**Adopter — Emma (German digital nomad, 29)**
Lives in a villa with garden, works from home, been on the island 6 months, wants a medium-sized calm companion. Speaks English and German. Found the shelter on Instagram.

**Adopter — Somchai (Thai local, 45)**
Has a house with a yard, had dogs before, wants to adopt a dog that's good with kids. Speaks Thai. Heard from a friend.

**Donor — Jake (Australian tourist, 32)**
Visiting Koh Phangan for 2 weeks, loves dogs but can't adopt. Wants to help financially. Speaks English.

**Shelter Admin — The Shelter Guy**
Runs everything from his phone. Needs to know when someone is interested, when a visit is booked, and nothing more. Does not want to learn new software.

---

## 3. Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Telegram    │────▶│   FastAPI         │────▶│  Claude API         │
│  Bot API     │◀────│   Backend         │◀────│  (Anthropic SDK)    │
└─────────────┘     └──────┬───────────┘     └─────────────────────┘
                           │                          │
                           │                     Tool Use
                           │                          │
                    ┌──────▼───────────────────────────▼──┐
                    │          Supabase                     │
                    │  ┌───────────┐  ┌─────────────────┐  │
                    │  │ Postgres  │  │ pgvector         │  │
                    │  │ (dogs,    │  │ (conversation    │  │
                    │  │  users,   │  │  memory)         │  │
                    │  │  bookings)│  │                  │  │
                    │  └───────────┘  └─────────────────┘  │
                    │  ┌───────────────────────────────┐   │
                    │  │ Storage (dog photos)           │   │
                    │  └───────────────────────────────┘   │
                    └──────────────────────────────────────┘
                           │
                    ┌──────▼──────────┐
                    │ Google Calendar  │
                    │ (via gws CLI)    │
                    └─────────────────┘
```

---

## 4. Tech Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Interface | Telegram Bot API (`python-telegram-bot` v20+) | Simple, popular in Thailand, mobile-first |
| Backend | FastAPI + Uvicorn | Async, lightweight, fast to build |
| AI Agent | Anthropic SDK (raw tool use loop) | Full control, maximum learning |
| Database | Supabase (Postgres + pgvector) | Structured data + vector memory in one service |
| File Storage | Supabase Storage | Dog photos, same platform |
| Image Processing | Pillow (PIL) | Photo optimization and thumbnail generation |
| Calendar | Google Calendar via `gws` CLI | Shelter visit scheduling |
| Notifications | Telegram API (direct message to admin) | Admin already uses Telegram |
| Language | Python 3.11+ | Single language for entire backend |

---

## 5. Database Schema

### 5.1 `dogs`

| Column | Type | Description |
|--------|------|-------------|
| id | UUID (PK) | Auto-generated |
| name | TEXT | Dog's name |
| breed | TEXT | Breed or mix description |
| age_estimate | TEXT | e.g. "~2 years", "puppy (3 months)" |
| size | TEXT | small, medium, large |
| gender | TEXT | male, female, unknown |
| temperament | TEXT[] | Array: calm, energetic, good_with_kids, etc. |
| medical_notes | TEXT | Vaccinations, spay/neuter, conditions |
| story | TEXT | How they were found, personality quirks |
| photos | TEXT[] | Supabase Storage URLs (optimized: max 1280px, JPEG, <300KB) |
| thumbnails | TEXT[] | Supabase Storage URLs (320px wide, JPEG, <50KB) |
| status | TEXT | available, reserved, adopted |
| intake_date | TIMESTAMP | When the dog arrived |
| instagram_post_text | TEXT | Auto-generated caption |
| created_at | TIMESTAMP | Record creation |
| updated_at | TIMESTAMP | Last update |

### 5.2 `users`

| Column | Type | Description |
|--------|------|-------------|
| id | UUID (PK) | Auto-generated |
| telegram_id | BIGINT (UNIQUE) | Telegram user identifier |
| telegram_username | TEXT | @username if available |
| name | TEXT | Extracted from conversation |
| language | TEXT | Detected language code |
| living_situation | TEXT | Villa, apartment, etc. |
| location | TEXT | Where they are / how long staying |
| experience_with_dogs | TEXT | First-time, experienced, etc. |
| lifestyle_notes | TEXT | Work from home, travels, kids, etc. |
| preferences | JSONB | Size, energy, gender preferences |
| funnel_stage | TEXT | curious, exploring, interested, ready, adopted, donor |
| liked_dog_ids | UUID[] | Dogs they showed interest in |
| intent | TEXT | adopt, donate, both, unknown |
| created_at | TIMESTAMP | First interaction |
| updated_at | TIMESTAMP | Last interaction |

### 5.3 `conversations`

| Column | Type | Description |
|--------|------|-------------|
| id | UUID (PK) | Auto-generated |
| user_id | UUID (FK → users) | Which user |
| summary | TEXT | AI-generated conversation summary |
| extracted_facts | JSONB | Structured facts from this conversation |
| embedding | VECTOR(1536) | For semantic search |
| messages_count | INT | Number of messages in this conversation |
| created_at | TIMESTAMP | When conversation occurred |

### 5.4 `bookings`

| Column | Type | Description |
|--------|------|-------------|
| id | UUID (PK) | Auto-generated |
| user_id | UUID (FK → users) | Adopter |
| dog_id | UUID (FK → dogs) | Dog being visited |
| scheduled_at | TIMESTAMP | Visit date/time |
| google_calendar_event_id | TEXT | Calendar reference |
| status | TEXT | scheduled, completed, cancelled, no_show |
| admin_notified | BOOLEAN | Whether admin was pinged |
| created_at | TIMESTAMP | Record creation |

---

## 6. Agent Design

### 6.1 Architecture

Single agent with tool use. No sub-agents, no multi-agent framework. One Claude instance with a system prompt and tools.

### 6.2 Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `search_dogs` | Query dogs by traits (size, temperament, etc.). Returns thumbnails. | Recommending dogs to adopter |
| `get_dog_profile` | Full profile + full-size photos for one dog | User asks about a specific dog |
| `get_user_profile` | Load known info about this user | Start of every conversation |
| `update_user_profile` | Save new info learned in conversation | After learning new facts |
| `recall_past_conversations` | Semantic search of conversation memory | Returning users, context recall |
| `check_calendar_availability` | Available shelter visit slots | User wants to visit |
| `book_shelter_visit` | Create booking + calendar event | Confirmed visit |
| `notify_admin` | Telegram message to shelter admin | Adoption interest, visits booked |
| `send_donation_info` | Share donation links | User wants to donate |

### 6.3 Agent Conversation Flow

```
User sends message
       │
       ▼
┌─────────────────────────┐
│ Load user profile        │ ◀── Always first (get_user_profile)
│ Load conversation memory │ ◀── If returning user (recall_past_conversations)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Claude API call          │
│ (system prompt + tools   │
│  + context + message)    │
└──────────┬──────────────┘
           │
     ┌─────▼─────┐
     │ Tool use?  │
     └─────┬─────┘
       YES │        NO
           ▼         ▼
     ┌──────────┐  ┌──────────────┐
     │ Execute   │  │ Return text  │
     │ tool(s)   │  │ to user      │
     └─────┬────┘  └──────┬───────┘
           │              │
           ▼              │
     ┌──────────┐         │
     │ Feed      │         │
     │ result    │         │
     │ back to   │──┐      │
     │ Claude    │  │      │
     └──────────┘  │      │
           ▲       │      │
           └───────┘      │
        (loop until       │
         text response)   │
                          │
                          ▼
                 ┌────────────────┐
                 │ Post-response:  │
                 │ Update profile  │
                 │ Save memory     │
                 └────────────────┘
```

### 6.4 Matching Philosophy

A good match considers:
- **Living situation:** Apartment vs house vs villa with garden
- **Lifestyle:** Active vs sedentary, remote work vs office, social vs quiet
- **Experience:** First-time owners need forgiving, patient dogs
- **Family:** Kids, other pets, partner preferences
- **Stay duration:** Permanent resident vs nomad vs tourist (tourists → redirect to donation)
- **Energy match:** High-energy person ↔ high-energy dog, calm ↔ calm

### 6.5 Funnel Stages

| Stage | Description | Agent Behavior |
|-------|-------------|---------------|
| `curious` | Just browsing, vague interest | Be welcoming, share cute dogs, no pressure |
| `exploring` | Asking about specific dogs | Provide detailed profiles, ask lifestyle questions |
| `interested` | Drawn to 1-2 dogs, practical questions | Answer concerns, address objections, suggest visit |
| `ready` | Wants to visit or adopt | Book visit, confirm details, notify admin |
| `adopted` | Completed adoption | Celebrate, offer post-adoption tips |
| `donor` | Chose to donate | Share links, express gratitude, provide updates |

### 6.6 System Prompt

See `src/agent/system_prompt.py`. Key elements:
- Warm, genuine personality — not salesy
- Knows every dog personally
- Responds in the user's language automatically
- Loads user profile at conversation start
- Tracks funnel progression naturally
- Handles objections with empathy and honesty
- Never pressures — trust leads to more adoptions

---

## 7. Features — Hackathon Scope

### 7.1 Dog Intake + Profile Generation (P0)

**As** the shelter admin,
**I want** to add a dog to the system with basic info and photos,
**So that** the agent can recommend them to potential adopters.

Acceptance criteria:
- Dog is stored in Supabase with all fields
- Photos are optimized before storage (max 1280px, JPEG, quality=85, EXIF stripped, <300KB)
- Thumbnails are generated (320px wide, quality=70, <50KB)
- Both full photos and thumbnails uploaded to Supabase Storage with public URLs
- Admin can send dog photos directly via Telegram message to the bot
- Instagram caption is auto-generated
- Dog appears in search results immediately

### 7.2 Conversational Matching Agent (P0)

**As** a potential adopter messaging the Telegram bot,
**I want** to have a natural conversation about my lifestyle and preferences,
**So that** I get personalized dog recommendations that fit my situation.

Acceptance criteria:
- Agent asks about lifestyle naturally (not as a form)
- Agent uses `search_dogs` to find matches
- Agent presents dogs with personalized pitches
- Agent shares dog thumbnails in search results, full photos for specific dog profiles
- Multiple photos sent as Telegram album (send_media_group)
- Agent handles multiple languages
- Agent remembers user across sessions

### 7.3 Shelter Visit Booking (P0)

**As** a potential adopter who wants to meet a dog,
**I want** to book a visit at the shelter,
**So that** I can meet the dog in person.

Acceptance criteria:
- Agent checks Google Calendar for available slots
- Agent confirms date/time with user
- Booking is created in database + Google Calendar
- Shelter admin is notified via Telegram with adopter info and dog name

### 7.4 Donation Routing (P0)

**As** someone who wants to support the shelter,
**I want** to receive the right donation link,
**So that** I can contribute financially.

Acceptance criteria:
- Agent detects donation intent from conversation
- One-time donors get GoFundMe link
- Recurring donors get subscription link (Patreon, Buy Me a Coffee, etc.)
- Agent thanks them and offers to share dog updates

### 7.5 User Memory + Profile (P0)

**As** a returning user,
**I want** the bot to remember who I am and what we discussed,
**So that** I don't have to repeat myself.

Acceptance criteria:
- User profile is created on first interaction
- Profile is updated after each conversation with new facts
- Conversation summaries are stored with embeddings
- On return visit, agent recalls preferences, liked dogs, funnel stage
- Agent references past interactions naturally

---

## 7.6 Photo Handling (P0)

**As** the shelter admin,
**I want** to send dog photos via Telegram and have them automatically optimized and stored,
**So that** adopters see great-looking photos that load fast.

**As** a potential adopter,
**I want** to see photos of recommended dogs directly in the chat,
**So that** I can feel a connection before visiting.

### Photo Pipeline

```
Admin sends photo via Telegram
       │
       ▼
┌─────────────────────────┐
│ Download from Telegram   │
│ (get highest resolution) │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Optimize with Pillow     │
│ • Resize: max 1280px     │
│ • Format: JPEG            │
│ • Quality: 85             │
│ • Strip EXIF              │
│ • Target: < 300KB         │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Generate thumbnail       │
│ • Width: 320px            │
│ • Quality: 70             │
│ • Target: < 50KB          │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Upload to Supabase       │
│ Storage (public bucket)  │
│ • dogs/{id}/photo_N.jpg  │
│ • dogs/{id}/thumb_N.jpg  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Update dog record with   │
│ photo + thumbnail URLs   │
└─────────────────────────┘
```

### Supabase Storage Setup

- Bucket name: `dog-photos`
- Public access: enabled (so Telegram can fetch URLs directly)
- Path structure: `dogs/{dog_id}/photo_{n}.jpg` and `dogs/{dog_id}/thumb_{n}.jpg`
- No authentication needed for reads — photos are public content

### Sending Photos to Adopters

- **Search results (multiple dogs):** Send thumbnails with dog name + one-line pitch
- **Specific dog profile:** Send full-size photos as a Telegram album (`send_media_group`)
- **Max 10 photos per album** (Telegram limit)
- If a dog has no photos, the agent should mention it naturally and still pitch the dog

---

## 8. Features — Post-Hackathon

### 8.1 Instagram DM Integration (P1)
Connect the same agent to Instagram Messaging API so adopters can chat via Instagram DMs.

### 8.2 Admin Dashboard (P1)
Next.js web app for the shelter admin:
- Dog management (add, edit, update status)
- Adoption pipeline (who's interested in which dog)
- Engagement statistics
- Donor tracking

### 8.3 Dog Status Updates (P2)
Adopters and donors can follow specific dogs and receive periodic updates (photos, videos, health status) posted by the shelter admin through the dashboard.

### 8.4 Auto-Generated Instagram Posts (P2)
On new dog intake, automatically generate and optionally publish an Instagram post with photo and caption.

### 8.5 Multi-Shelter Support (P3)
Extend the platform to support multiple shelters across different islands/locations.

### 8.6 Dog Sponsorship Model (P3)
"Sponsor a dog" — monthly recurring donation tied to a specific dog, with regular progress updates.

---

## 9. Non-Functional Requirements

- **Response time:** Agent should respond within 5 seconds for simple messages, 15 seconds max for tool-heavy responses
- **Languages:** Must handle English, Thai, German, French, Spanish at minimum (Claude handles this natively)
- **Availability:** Bot should run 24/7 (deploy on Railway or Fly.io)
- **Data privacy:** Don't store unnecessary personal info, be transparent about what's saved
- **Error resilience:** Bot should never crash from a user message. All errors handled gracefully.

---

## 10. Sample Dog Data (for hackathon seeding)

| Name | Breed | Age | Size | Temperament | Story |
|------|-------|-----|------|-------------|-------|
| Mango | Thai Ridgeback mix | ~2 years | Medium | Calm, loyal, gentle | Found wandering near Thong Sala pier, extremely sweet and well-behaved |
| Coconut | Mixed breed | ~6 months | Small | Playful, curious, energetic | Rescued as a tiny puppy from a construction site, full of joy |
| Luna | Lab mix | ~3 years | Large | Friendly, good with kids, social | Abandoned when her family left the island, still trusts everyone |
| Diesel | Thai Bangkaew mix | ~4 years | Large | Protective, loyal, smart | Street dog who chose the shelter, guards it like it's his home |
| Noodle | Mixed breed | ~1 year | Small | Shy, sweet, cuddly | Found injured and scared, slowly becoming the most affectionate dog |
| Sunny | Golden mix | ~2 years | Medium | Happy, energetic, loves water | Found on the beach, lives for swimming and fetch |
| Ghost | White mixed breed | ~5 years | Medium | Calm, independent, quiet | Senior boy who's been at the shelter longest, dignified and gentle |
| Pepper | Mixed breed | ~8 months | Small | Bold, clever, mischievous | Tiny but fearless, escapes every enclosure, steals everyone's food |

---

## 11. Success Metrics

For the hackathon demo:
- Agent successfully matches a user to a dog based on conversation
- Agent remembers a returning user's preferences
- Shelter visit is booked on Google Calendar
- Admin receives notification
- Agent works in at least 2 languages in the same demo

Long-term:
- Number of adoptions facilitated
- Average time from first conversation to shelter visit
- Donor conversion rate
- User return rate (people coming back to chat)
