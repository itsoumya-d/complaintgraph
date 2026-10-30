# ComplaintGraph — Intercom Conversations

*Generated 2026-10-30 04:19 UTC from 284 public reviews. This is an independent analysis, not an official company document.*

## Snapshot

- Reviews analysed: **284**
- Star-derived sentiment: **negative 166**, neutral 42, positive 76
- Star distribution: 1★ 120, 2★ 46, 3★ 42, 4★ 28, 5★ 48

## Model

- Base: `distilbert-base-uncased`
- Eval accuracy: **0.690**, macro F1: **0.689**
- Train/eval size: 2880/720

## What people are complaining about

| # | Theme | Reviews | Likely surface | Top terms |
|---|---|---:|---|---|
| 1 | Mobile conversation workflow | 87 | Mobile conversation workflow | conversation, mobile, does, time, chat, notification |
| 2 | Sign-in & account access | 27 | Onboarding / authentication | login, open, password, time, email, account |
| 3 | Notifications & alerts | 19 | Notifications | notifications, push notifications, push, settings, receiving, receiving notifications |
| 4 | Update & release stability | 18 | Reliability / performance | update, android, version, broken, crashing, latest |
| 5 | Sign-in & account access | 15 | Onboarding / authentication | google, google account, account, sign, login, email |

### Theme 1 — Mobile conversation workflow

**Evidence (87 reviews):**
> Not User Friendly I wanted intercom to be mobile friendly so I could follow up with my clients anywhere, any time, hence I downloaded this app, but it just seems so disappointing. - Dark Mode is alright, but it's so hard to differentiate if you're on note, or conversation with the client. - there's 
> Clunky, slow app which crashes all the time. Conversations do not load just the loader keeps showing. Cannot upload files either. Keeps logging me out all the time. Can't even use the website on the mobile browser since it isn't optimized at all. Please fix it.
> Why can a ticket/Conversation not be started from the Mobile app? Every other competitor allows this. Gorgias, Richpanel, Tidio. But you cannot start a new conversation from the Intercom mobile app. Dumb

**Suggested experiment:** Profile the conversation list and thread load, add skeleton/retry states, offline drafts and push-to-thread; measure time-to-first-conversation.

### Theme 2 — Sign-in & account access · Onboarding / authentication

**Evidence (27 reviews):**
> When you open this app, the options are to enter a login and password, or login to an existing account with Google. If there was an option to create an account this review might be longer, but I have no experience beyond the login screen.
> Won't even open
> Can't event login or register an account.

**Suggested experiment:** Instrument OAuth/OTP failure reasons, add a fallback sign-in path, and measure login success rate before/after.

### Theme 3 — Notifications & alerts · Notifications

**Evidence (19 reviews):**
> Notifications continuously repeat regardless of setting push notifications off, vibrations off, sounds off.
> Please fix push notifications, I don't get any even though they are all on in settings. App is useless without them
> A suggestion. It'd be nice if there was an option to change the notification sound. I need to pull my phone out only on Intercom notifications. Plus, even if I turn the notification sound off, the app sounds while showing notification. However, I turned app notifications off. Now, I've turned it bac

**Suggested experiment:** Audit notification fan-out and dedupe, add per-conversation mute/clear, and measure opt-outs and notification-driven opens.

### Theme 4 — Update & release stability · Reliability / performance

**Evidence (18 reviews):**
> Khankir chele update
> latest update on Android broken
> Won't open after update, crashing after tap on icon, not working on my android 11 phone

**Suggested experiment:** Gate the change behind a staged rollout, track crash-free sessions, and keep a one-tap rollback path.

### Theme 5 — Sign-in & account access · Onboarding / authentication

**Evidence (15 reviews):**
> Not working with Google account
> Still can't connect with my Google account
> Doesn't sign in with google account

**Suggested experiment:** Instrument OAuth/OTP failure reasons, add a fallback sign-in path, and measure login success rate before/after.

## Method

1. Public reviews are labelled from their star rating (1–2 negative, 3 neutral, 4–5 positive).
2. A small transformer (`prajjwal1/bert-tiny`) is fine-tuned on a balanced public sample.
3. Negative reviews are clustered with TF-IDF + KMeans; each cluster is represented by top terms and nearest-neighbour quotes.
4. The report is generated from those clusters — no private data and no proprietary source code.

## Limitations

- Star ratings are a noisy proxy for sentiment; review text can be sarcastic or mixed.
- The model is a lightweight baseline, not a production classifier. Treat the scores as directional.
- Clusters are descriptive, not causal. The suggested experiments are hypotheses to validate.
- Public reviews over-represent highly motivated users; pair this with support tickets and product analytics before deciding.
