from crewai import Task
from textwrap import dedent
from datetime import date


class TripTasks():

  def identify_task(self, agent, origin, cities, interests, range):
    return Task(description=dedent(f"""
        Analyze and select the best city for the trip based
        on specific criteria such as weather patterns, seasonal
        events, and travel costs. This task involves comparing
        multiple cities, considering factors like current weather
        conditions, upcoming cultural or seasonal events, and
        overall travel expenses.

        Traveling from: {origin}
        City Options: {cities}
        Trip Date: {range}
        Traveler Interests: {interests}
      """),
              agent=agent,
              expected_output=dedent(f"""Your final answer must be a detailed
        report on the chosen city, and everything you found out
        about it, including the actual flight costs, weather
        forecast and attractions.
       """))

  def gather_task(self, agent, origin, interests, range, context=None):
    return Task(description=dedent(f"""
        As a local expert on this city you must compile an
        in-depth guide for someone traveling there and wanting
        to have THE BEST trip ever!
        Gather information about  key attractions, local customs,
        special events, and daily activity recommendations.
        Find the best spots to go to, the kind of place only a
        local would know.
        This guide should provide a thorough overview of what
        the city has to offer, including hidden gems, cultural
        hotspots, must-visit landmarks, weather forecasts, and
        high level costs.

        Trip Date: {range}
        Traveling from: {origin}
        Traveler Interests: {interests}
      """),
                agent=agent,
                context=context,
                expected_output=dedent(f"""The final answer must be a comprehensive city guide,
        rich in cultural insights and practical tips,
        tailored to enhance the travel experience.
       """))

  def plan_task(self, agent, origin, interests, range, context=None):
    return Task(description=dedent(f"""
        Expand this guide into a a full 7-day travel
        itinerary with detailed per-day plans, including
        weather forecasts, places to eat, packing suggestions,
        and a budget breakdown.

        You MUST suggest actual places to visit, actual hotels
        to stay and actual restaurants to go to.

        This itinerary should cover all aspects of the trip,
        from arrival to departure, integrating the city guide
        information with practical travel logistics.

        Trip Date: {range}
        Traveling from: {origin}
        Traveler Interests: {interests}
      """),
                agent=agent,
                context=context,
                expected_output=dedent(f"""Your final answer MUST be a complete expanded travel plan,
        formatted as markdown, encompassing a daily schedule,
        anticipated weather conditions, recommended clothing and
        items to pack, and a detailed budget, ensuring THE BEST
        TRIP EVER, Be specific and give it a reason why you picked
        # up each place, what make them special!"""))

