-- Doggies — Database Schema
-- Run this in the Supabase SQL editor to set up the database.

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------

create extension if not exists "uuid-ossp";
create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- dogs
-- ---------------------------------------------------------------------------

create table if not exists dogs (
    id              uuid primary key default uuid_generate_v4(),
    name            text not null,
    breed           text not null,
    age_estimate    text not null,
    size            text not null check (size in ('small', 'medium', 'large')),
    gender          text not null check (gender in ('male', 'female', 'unknown')),
    temperament     text[] not null default '{}',
    medical_notes   text,
    story           text,
    photos          text[] not null default '{}',
    thumbnails      text[] not null default '{}',
    status          text not null default 'available' check (status in ('available', 'reserved', 'adopted')),
    intake_date     timestamptz,
    instagram_post_text text,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create index if not exists dogs_status_idx on dogs (status);
create index if not exists dogs_size_idx on dogs (size);

-- Auto-update updated_at
create or replace function update_updated_at_column()
returns trigger language plpgsql as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create or replace trigger dogs_updated_at
    before update on dogs
    for each row execute function update_updated_at_column();

-- ---------------------------------------------------------------------------
-- users
-- ---------------------------------------------------------------------------

create table if not exists users (
    id                    uuid primary key default uuid_generate_v4(),
    telegram_id           bigint not null unique,
    telegram_username     text,
    name                  text,
    language              text,
    living_situation      text,
    location              text,
    experience_with_dogs  text,
    lifestyle_notes       text,
    preferences           jsonb not null default '{}',
    funnel_stage          text not null default 'curious'
                            check (funnel_stage in ('curious', 'exploring', 'interested', 'ready', 'adopted', 'donor')),
    liked_dog_ids         uuid[] not null default '{}',
    intent                text not null default 'unknown'
                            check (intent in ('adopt', 'donate', 'both', 'unknown')),
    created_at            timestamptz not null default now(),
    updated_at            timestamptz not null default now()
);

create index if not exists users_telegram_id_idx on users (telegram_id);

create or replace trigger users_updated_at
    before update on users
    for each row execute function update_updated_at_column();

-- ---------------------------------------------------------------------------
-- conversations
-- ---------------------------------------------------------------------------

create table if not exists conversations (
    id              uuid primary key default uuid_generate_v4(),
    user_id         uuid not null references users (id) on delete cascade,
    summary         text not null,
    extracted_facts jsonb not null default '{}',
    embedding       vector(1536),
    messages_count  int not null default 0,
    created_at      timestamptz not null default now()
);

create index if not exists conversations_user_id_idx on conversations (user_id);
create index if not exists conversations_embedding_idx
    on conversations using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- Semantic similarity search function used by the memory module.
-- Returns conversations ordered by cosine distance from query_embedding.
create or replace function match_conversations(
    query_embedding  vector(1536),
    match_user_id    uuid,
    match_count      int default 5
)
returns table (
    id              uuid,
    user_id         uuid,
    summary         text,
    extracted_facts jsonb,
    messages_count  int,
    created_at      timestamptz,
    similarity      float
)
language sql stable as $$
    select
        c.id,
        c.user_id,
        c.summary,
        c.extracted_facts,
        c.messages_count,
        c.created_at,
        1 - (c.embedding <=> query_embedding) as similarity
    from conversations c
    where c.user_id = match_user_id
      and c.embedding is not null
    order by c.embedding <=> query_embedding
    limit match_count;
$$;

-- ---------------------------------------------------------------------------
-- bookings
-- ---------------------------------------------------------------------------

create table if not exists bookings (
    id                        uuid primary key default uuid_generate_v4(),
    user_id                   uuid not null references users (id) on delete cascade,
    dog_id                    uuid not null references dogs (id) on delete cascade,
    scheduled_at              timestamptz not null,
    google_calendar_event_id  text,
    status                    text not null default 'scheduled'
                                check (status in ('scheduled', 'completed', 'cancelled', 'no_show')),
    admin_notified            boolean not null default false,
    created_at                timestamptz not null default now()
);

create index if not exists bookings_user_id_idx on bookings (user_id);
create index if not exists bookings_dog_id_idx on bookings (dog_id);
create index if not exists bookings_scheduled_at_idx on bookings (scheduled_at);

-- ---------------------------------------------------------------------------
-- Supabase Storage: dog-photos bucket
-- Run this separately in the Storage section or via the Supabase dashboard.
-- ---------------------------------------------------------------------------
-- insert into storage.buckets (id, name, public)
-- values ('dog-photos', 'dog-photos', true)
-- on conflict do nothing;
